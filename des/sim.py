import heapq
import inspect
import types
import warnings
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from random import Random
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Generator,
    Iterable,
    Mapping,
    Protocol,
    Self,
    Sequence,
    assert_never,
    cast,
)
from uuid import UUID, uuid4

from frozendict import frozendict

if TYPE_CHECKING:
    from _typeshed import SupportsAllComparisons

__all__ = [
    "Message",
    "message",
    "Addr",
    "Ref",
    "Atom",
    "RunResult",
    "SMBuilder",
    "StateMachine",
    "StateMachineInit",
    "EventLoop",
    "loop",
    "launch",
    "send",
    "self",
    "_self_node",
    "wait",
    "log",
    "now",
    "rng",
    "ask",
    "ask_timeout",
    "Timeout",
    "sleep",
    "sleep_until",
]


@dataclass(frozen=True, slots=True)
class Message:
    args: tuple[Any, ...]
    kwargs: frozendict[str, Any]

    def __repr__(self) -> str:
        args_strs = (repr(arg) for arg in self.args)
        kwargs_strs = (f"{k}={v!r}" for k, v in self.kwargs.items())
        return f"{type(self).__name__}({', '.join((*args_strs, *kwargs_strs))})"


def message(*args: Any, **kwargs: Any) -> Message:
    return Message(args, frozendict(kwargs))


@dataclass(frozen=True, slots=True)
class Addr:
    name: str

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


def _deterministic_uuid() -> UUID:
    if _context_var.get() is None:
        warnings.warn("Ref created outside of a scoped context; using random UUID")
        return uuid4()
    else:
        return UUID(bytes=rng().randbytes(16))


def created_by_default() -> Addr | None:
    try:
        return context().self_addr
    except RuntimeError:
        return None


@dataclass(frozen=True, slots=True)
class Ref:
    uuid: UUID = field(default_factory=_deterministic_uuid)
    # created_by: Addr | None = field(default_factory=created_by_default)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({str(self.uuid)!r})"


class _AtomMeta(type):
    def __str__(cls) -> str:
        return cls.__name__

    def __repr__(cls) -> str:
        return str(cls)

    def __instancecheck__(self, instance: Any, /) -> bool:
        return type(instance) is self or instance is self


class Atom(metaclass=_AtomMeta):
    pass


@dataclass
class MatchResult:
    matched: bool
    """If the message matches"""
    branch_id: Any = None
    """Identifier for the matched branch"""
    matched_args: set[int] = field(default_factory=set)
    """Indices of matched positional arguments"""
    matched_kwargs: set[str] = field(default_factory=set)
    """Keys of matched keyword arguments"""

    def __bool__(self):
        return self.matched


class Matcher(Protocol):
    def match(self, msg: Message) -> MatchResult: ...


class BranchIDMatcher(Matcher):
    def __init__(self, inner: Matcher, branch_id: Any):
        self.inner = inner
        self.branch_id = branch_id

    def match(self, msg: Message) -> MatchResult:
        return replace(self.inner.match(msg), branch_id=self.branch_id)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(inner={self.inner!r}, branch_id={self.branch_id!r})"
        )


class SingleMatcher(Matcher):
    def __init__(
        self, args: Sequence[Any], kwargs: Mapping[str, Any], branch_id: Any = None
    ):
        self.args = args
        self.kwargs = kwargs
        self.branch_id = branch_id

    def match(self, msg: Message) -> MatchResult:
        matched_args = set()
        matched_kwargs = set()

        for i, (actual, expected) in enumerate(zip(msg.args, self.args)):
            if expected is Ellipsis:
                continue
            if actual == expected:
                matched_args.add(i)
            else:
                return MatchResult(matched=False)

        for key, expected in self.kwargs.items():
            if key in msg.kwargs and msg.kwargs[key] == expected:
                matched_kwargs.add(key)
            else:
                return MatchResult(matched=False)

        return MatchResult(
            matched=True, matched_args=matched_args, matched_kwargs=matched_kwargs
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(args={self.args!r}, kwargs={self.kwargs!r}, "
            f"branch_id={self.branch_id!r})"
        )


class SelectionMatcher(Matcher):
    def __init__(self, *options: Matcher):
        self.options = options

    def match(self, msg: Message) -> MatchResult:
        for option in self.options:
            result = option.match(msg)
            if result.matched:
                return result
        return MatchResult(matched=False)

    def __repr__(self) -> str:
        options_str = ", ".join(repr(option) for option in self.options)
        return f"{type(self).__name__}({options_str})"


class Context(Protocol):
    @property
    def self_addr(self) -> Addr: ...
    @property
    def sent_messages(self) -> list[Event]: ...
    @property
    def self_node(self) -> Process: ...
    def current_epoch(self) -> int: ...
    def append_log(self, log: LogEntry): ...


class StateMachine(Protocol):
    @property
    def matcher(self) -> Matcher: ...
    def next(self, ctx: Context, match: MatchResult, msg: Message) -> StateMachine: ...

    @property
    def sm(self) -> Self: ...

    @property
    def messages(self) -> Sequence[Message]: ...


def next_state(
    sm: StateMachine, ctx: Context, msgs: Sequence[Message]
) -> tuple[int, StateMachine] | None:
    for i, msg in enumerate(msgs):
        if match := sm.matcher.match(msg):
            next_sm = sm.next(ctx, match, msg)
            return i, next_sm
    return None


class StateMachineInit(Protocol):
    @property
    def sm(self) -> StateMachine: ...

    @property
    def messages(self) -> Sequence[Message]: ...


class NoOpInit:
    @property
    def sm(self) -> Self:
        return self

    @property
    def messages(self) -> Sequence[Message]:
        return ()


@dataclass
class StateMachineInitImpl:
    sm: StateMachine
    messages: Sequence[Message] = ()


def initialize(sm: StateMachine, *msgs: Message) -> StateMachineInit:
    return StateMachineInitImpl(sm, msgs)


@dataclass
class ResumeSM:
    match: MatchResult
    msg: Message


def _filter_message(msg: Message, match: MatchResult) -> Message:
    filtered_args = tuple(
        arg for i, arg in enumerate(msg.args) if i not in match.matched_args
    )
    filtered_kwargs = {
        key: value
        for key, value in msg.kwargs.items()
        if key not in match.matched_kwargs
    }
    return Message(filtered_args, frozendict(filtered_kwargs))


type SMCoroutine = Coroutine[Matcher, ResumeSM | None, None]


class StopStateMachine(Exception):
    pass


class CoroutineSM(NoOpInit):
    def __init__(self, co: SMCoroutine, continuation: StateMachine):
        self.co = co
        try:
            self.matcher = co.send(None)
        except StopIteration:
            raise StopStateMachine()
        self.continuation = continuation

    @classmethod
    def create(cls, co: SMCoroutine, continuation: StateMachine) -> StateMachine:
        try:
            return cls(co, continuation)
        except StopStateMachine:
            return continuation

    def next(self, ctx: Context, match: MatchResult, msg: Message) -> StateMachine:
        try:
            with scoped_context(ctx):
                next_matcher = self.co.send(ResumeSM(match=match, msg=msg))
            self.matcher = next_matcher
            return self
        except StopIteration:
            return self.continuation


class SMBuilderTransition(Protocol):
    def __call__(self, msg: Message) -> StateMachine: ...


class SMBuilder(NoOpInit):
    def __init__(self):
        self.branches = list[tuple[Matcher, SMBuilderTransition]]()

    @property
    def matcher(self) -> Matcher:
        return SelectionMatcher(
            *(
                BranchIDMatcher(matcher, branch_id=i)
                for i, (matcher, _) in enumerate(self.branches)
            )
        )

    def next(self, ctx: Context, match: MatchResult, msg: Message) -> StateMachine:
        transition = self.branches[cast(int, match.branch_id)][1]
        with scoped_context(ctx):
            return transition(_filter_message(msg, match))

    def add_branch(self, matcher: Matcher, transition: SMBuilderTransition):
        self.branches.append((matcher, transition))

    def handle(self, *match_args: Any, **match_kwargs: Any):
        def decorator(fn: Callable[..., SMCoroutine] | Callable[..., StateMachine]):
            if inspect.iscoroutinefunction(fn):
                self.add_branch(
                    matcher=SingleMatcher(match_args, match_kwargs),
                    transition=async_transition(continuation=self)(handler(fn)),
                )
            elif inspect.isfunction(fn):
                self.add_branch(
                    matcher=SingleMatcher(match_args, match_kwargs),
                    transition=handler(fn),
                )
            else:
                raise TypeError("Expected a function or coroutine function")

        return decorator


class HandlerFn[T](Protocol):
    def __call__(self, msg: Message) -> T: ...


_context_var = ContextVar[Context | None]("context")


def context() -> Context:
    ctx = _context_var.get()
    if ctx is None:
        raise RuntimeError("context() called outside of a scoped context")
    return ctx


@contextmanager
def scoped_context(ctx: Context):
    token = _context_var.set(ctx)
    try:
        yield
    finally:
        _context_var.reset(token)


def handler[T](fn: Callable[..., T]) -> HandlerFn[T]:
    sig = inspect.signature(fn)

    def wrapped(msg: Message) -> T:
        bound = sig.bind(
            *msg.args, **{k: v for k, v in msg.kwargs.items() if k in sig.parameters}
        )
        result = fn(*bound.args, **bound.kwargs)
        return result

    wrapped.__name__ = fn.__name__

    return wrapped


class SMBuilderAsyncTransition(Protocol):
    def __call__(self, msg: Message) -> SMCoroutine: ...


def async_transition(continuation: StateMachine):
    def decorator(transition: SMBuilderAsyncTransition):
        def wrapped(msg: Message) -> StateMachine:
            return CoroutineSM.create(co=transition(msg), continuation=continuation)

        if hasattr(transition, "__name__"):
            wrapped.__name__ = getattr(transition, "__name__")

        return wrapped

    return decorator


@dataclass
class ContextImpl:
    _current_epoch: int
    _self_addr: Addr
    sent_messages: list[Event]
    self_node: Process
    logs: list[LogEntry] = field(default_factory=list)

    num_now_calls: int = 0
    num_self_calls: int = 0

    def current_epoch(self):
        self.num_now_calls += 1
        return self._current_epoch

    @property
    def self_addr(self) -> Addr:
        self.num_self_calls += 1
        return self._self_addr

    def append_log(self, log: LogEntry):
        self.logs.append(log)


class _LeakDetector:
    def __init__(self, alpha: float = 0.1, threshold: float = 0.01):
        self.ema_diff = 0
        self.prev_value = None
        self.alpha = alpha
        self.threshold = threshold

    def update(self, current_memory: int):
        if self.prev_value is not None:
            diff = current_memory - self.prev_value
            self.ema_diff = self.alpha * diff + (1 - self.alpha) * self.ema_diff
        self.prev_value = current_memory

    def is_leaking(self):
        return self.ema_diff > self.threshold


class StateMarker:
    uuid: UUID = field(default_factory=uuid4)


class CloneContext:
    def __init__(
        self,
        current_epoch_log: EffectLogRead[int],
        self_addr_log: EffectLogRead[Addr],
        self_node: Process,
    ):
        self.sent_messages: list[Event] = []
        self.current_epoch_log = current_epoch_log
        self.self_addr_log = self_addr_log
        self.self_node = self_node

    @property
    def self_addr(self) -> Addr:
        result = self.self_addr_log.pop()
        if result is None:
            raise RuntimeError("Inconsistent access pattern during cloning")
        self.self_node._self_addr_log.append(result)
        return result

    def current_epoch(self) -> int:
        result = self.current_epoch_log.pop()
        if result is None:
            raise RuntimeError("Inconsistent access pattern during cloning")
        self.self_node._current_epoch_log.append(result)
        return result

    def append_log(self, log: LogEntry):
        del log  # unused
        pass


class EffectLogWrite[T]:
    def __init__(self):
        self._entries: list[tuple[T, int]] = []

    def append(self, entry: T, times: int = 1):
        if times <= 0:
            return
        if self._entries and self._entries[-1][0] == entry:
            last_entry, last_times = self._entries[-1]
            self._entries[-1] = (last_entry, last_times + times)
        self._entries.append((entry, times))

    def to_reader(self) -> EffectLogRead[T]:
        reader = EffectLogRead[T]()
        reader._entries = deque(self._entries)
        return reader


class EffectLogRead[T]:
    def __init__(self):
        self._entries = deque[tuple[T, int]]()

    def pop(self) -> T | None:
        if not self._entries:
            return None
        entry, times = self._entries[0]
        if times > 1:
            self._entries[0] = (entry, times - 1)
        else:
            self._entries.popleft()
        return entry


class Process:
    def __init__(
        self, sm_factory: Callable[[], StateMachineInit], seed: int | str | bytes
    ):
        self._factory = sm_factory
        init = sm_factory()
        self._sm = init.sm
        self._inbox: list[Message] = [*init.messages]
        self._drop_hints: list[Matcher] = []
        self._stopped = False
        self._detector = _LeakDetector(alpha=0.09, threshold=0.1)
        self._init_seed = seed
        self._rng = Random(seed)
        self._event_log: list[Message | Seed | StateMarker] = []
        self._current_epoch_log = EffectLogWrite[int]()
        self._self_addr_log = EffectLogWrite[Addr]()
        self._log: list[LogEntry] = []

    def run(
        self, self_addr: Addr, event_loop: EventLoop
    ) -> tuple[list[Event], list[LogEntry]]:
        if self._stopped:
            return [], []
        ctx = ContextImpl(
            _current_epoch=event_loop.epoch,
            _self_addr=self_addr,
            sent_messages=[],
            self_node=self,
        )
        while True:
            result = next_state(self._sm, ctx=ctx, msgs=self._inbox)
            if result is not None:
                i, next_sm = result
                self._sm = next_sm
                msg = self._inbox.pop(i)
                self._event_log.append(msg)
            else:
                break
        self._analyze_inbox_growth(self_addr)
        self._current_epoch_log.append(event_loop.epoch, times=ctx.num_now_calls)
        self._self_addr_log.append(self_addr, times=ctx.num_self_calls)
        return ctx.sent_messages, ctx.logs

    def _analyze_inbox_growth(self, addr: Addr):
        self._detector.update(len(self._inbox))
        if self._detector.is_leaking():
            warnings.warn(f"{addr} inbox growing. size={len(self._inbox)}")
            print(self._inbox, self._sm.matcher)

    def seed_rng(self, seed: int | str | bytes):
        self._event_log.append(Seed(seed))
        self._rng.seed(seed)

    def deliver(self, msg: Message):
        if self._stopped:
            return
        for i, hint in enumerate(self._drop_hints):
            if hint.match(msg):
                self._drop_hints.pop(i)
                return
        self._inbox.append(msg)

    def set_drop_hint(self, matcher: Matcher):
        for i, msg in enumerate(self._inbox):
            if matcher.match(msg):
                self._inbox.pop(i)
                return
        self._drop_hints.append(matcher)

    def add_marker(self, marker: StateMarker | None = None) -> StateMarker:
        if marker is None:
            marker = StateMarker()
        self._event_log.append(marker)
        return marker

    def clone(self, marker: StateMarker | None = None) -> Process:
        new_process = Process(self._factory, seed=self._init_seed)
        new_process._inbox.clear()
        if marker is None:
            events = self._event_log
        else:
            try:
                index = self._event_log.index(marker) + 1
            except ValueError:
                raise ValueError("Marker not found in event log") from None
            events = self._event_log[:index]
        for event in events:
            match event:
                case Seed(data):
                    new_process.seed_rng(data)
                case Message():
                    if next := next_state(
                        new_process._sm,
                        ctx=CloneContext(
                            current_epoch_log=self._current_epoch_log.to_reader(),
                            self_addr_log=self._self_addr_log.to_reader(),
                            self_node=new_process,
                        ),
                        msgs=[event],
                    ):
                        _, next_sm = next
                        new_process._sm = next_sm
                    else:
                        raise RuntimeError(
                            "Inconsistent state during cloning: message could not be delivered"
                        )

                case StateMarker():
                    if event == marker:
                        break
                case _:
                    assert_never(event)
        new_process._event_log = events.copy()
        return new_process


@dataclass(frozen=True, slots=True)
class LogEntry:
    epoch: int
    addr: Addr
    data: dict[str, Any]

    @property
    def msg(self) -> str | None:
        return self.data.get("_msg")

    def __str__(self) -> str:
        if self.msg is not None:
            msg = f"{self.msg} "
        else:
            msg = ""
        data = " ".join(f"{k}={v!r}" for k, v in self.data.items() if k != "_msg")
        return f"[{self.epoch:06}] [{self.addr.name}] {msg}{data}"


@dataclass
class RunResult:
    logs: list[LogEntry]
    num_epochs: int


@dataclass(slots=True)
class _PQEntry[T]:
    priority: SupportsAllComparisons
    item: T

    def __lt__(self, other: Self):
        return self.priority < other.priority

    def __gt__(self, other: Self):
        return self.priority > other.priority


class PriorityQueue[T]:
    def __init__(self, *items: T, key: Callable[[T]] = lambda x: x):
        self._key = key
        self._items = list(_PQEntry(key(item), item) for item in items)
        heapq.heapify(self._items)

    def push(self, item: T):
        heapq.heappush(self._items, _PQEntry(self._key(item), item))

    def extend(self, items: Iterable[T]):
        for item in items:
            self.push(item)

    def pop(self) -> T | None:
        if not self._items:
            return None
        return heapq.heappop(self._items).item

    def pop_while(self, cond: Callable[[T], bool]) -> Generator[T]:
        while self._items and cond(self._items[0].item):
            yield heapq.heappop(self._items).item

    def peek(self) -> T | None:
        if not self._items:
            return None
        return self._items[0].item

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def copy(self) -> PriorityQueue[T]:
        """Return a shallow copy of the priority queue."""
        new_pq = PriorityQueue(key=self._key)
        new_pq._items = self._items.copy()
        return new_pq


@dataclass
class Seed:
    data: int | str | bytes


@dataclass
class Event:
    recipient: Addr
    delivery_epoch: int
    payload: Message | Seed


class EventLoop:
    def __init__(self, seed: int | str | None = None):
        self.nodes = dict[Addr, Process]()
        self.logs: list[LogEntry] = []
        self.epoch = 0
        self.event_queue = PriorityQueue[Event](key=lambda event: event.delivery_epoch)
        self.seed = str(seed) if seed else ""

    def spawn(
        self,
        addr: Addr,
        init: Callable[[], StateMachineInit],
    ):
        node = Process(init, seed=f"{addr.name}{self.seed}")
        self.nodes[addr] = node
        return node

    def run(self, epochs: int = 1):
        last_epoch = self.epoch
        while True:
            if self.epoch >= epochs:
                break

            events = list(
                self.event_queue.pop_while(lambda e: e.delivery_epoch <= self.epoch)
            )

            for event in events:
                try:
                    node = self.nodes[event.recipient]
                except KeyError:
                    raise RuntimeError(f"Send to unknown address: {event}")
                match event.payload:
                    case Seed(data):
                        node.seed_rng(data)
                    case Message():
                        node.deliver(event.payload)

            for addr, node in self.nodes.items():
                sent_msgs, new_logs = node.run(
                    self_addr=addr,
                    event_loop=self,
                )
                self.event_queue.extend(sent_msgs)
                self.logs.extend(new_logs)

            last_epoch = self.epoch

            if next_event := self.event_queue.peek():
                self.epoch = max(self.epoch + 1, next_event.delivery_epoch)
            else:
                break

        return RunResult(logs=self.logs, num_epochs=last_epoch)

    def clone(self) -> EventLoop:
        new_loop = EventLoop(seed=self.seed)
        new_loop.epoch = self.epoch
        new_loop.logs = self.logs.copy()
        new_loop.event_queue = self.event_queue.copy()
        new_loop.nodes = {addr: node.clone() for addr, node in self.nodes.items()}
        return new_loop

    def find(self, addr: Addr | str) -> Process:
        if isinstance(addr, str):
            addr = Addr(addr)
        try:
            return self.nodes[addr]
        except KeyError:
            raise KeyError(f"Unknown address: {addr}")


def send(addr: Addr, *args: Any, **kwargs: Any):
    context().sent_messages.append(
        Event(recipient=addr, delivery_epoch=0, payload=message(*args, **kwargs))
    )


def send_scheduled(at: int, addr: Addr, *args: Any, **kwargs: Any):
    context().sent_messages.append(
        Event(recipient=addr, delivery_epoch=at, payload=message(*args, **kwargs))
    )


def self() -> Addr:
    return context().self_addr


def _self_node() -> Process:
    return context().self_node


@types.coroutine
def _wait_inner(
    matcher: Matcher,
) -> Generator[Matcher, ResumeSM, tuple[MatchResult, Message]]:
    resume = yield matcher
    return resume.match, resume.msg


async def wait(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
    match, msg = await _wait_inner(SingleMatcher(args, kwargs))
    msg = _filter_message(msg, match)
    return *msg.args, msg.kwargs


async def _race_refs(*refs: Ref):
    result, message = await _wait_inner(_FusedRefSelect(refs))
    dropped = [ref for i, ref in enumerate(refs) if i != result.branch_id]
    for ref in dropped:
        _self_node().set_drop_hint(_FusedRefSelect((ref,)))
    return message


def log(msg: str | None = None, /, **kwargs: Any):
    if msg is not None:
        kwargs["_msg"] = msg
    context().append_log(LogEntry(epoch=now(), addr=self(), data=kwargs))


def now() -> int:
    return context().current_epoch()


def rng() -> Random:
    return _self_node()._rng


async def ask(addr: Addr, method: Any, *args: Any, **kwargs: Any) -> Any:
    ref = Ref()
    send(addr, method, self(), ref, *args, **kwargs)
    result, *_ = await wait(ref)
    return result


def stop():
    _self_node()._stopped = True


class _FusedRefSelect:
    def __init__(self, refs: tuple[Ref, ...]):
        self.refs = refs

    def match(self, msg: Message) -> MatchResult:
        for i, ref in enumerate(self.refs):
            if msg.args and msg.args[0] == ref:
                return MatchResult(matched=True, branch_id=i, matched_args={0})
        return MatchResult(matched=False)


@dataclass
class Timeout(TimeoutError):
    msg: str | None = None
    is_deadline: bool = False


async def ask_timeout(
    timeout: int,
    addr: Addr,
    method: Any,
    *args: Any,
) -> Any:
    result_ref = Ref()
    send(addr, method, self(), result_ref, *args)
    timeout_ref = Ref()
    send_scheduled(now() + timeout, self(), timeout_ref, hint="timeout")
    msg = await _race_refs(result_ref, timeout_ref)
    if msg.args[0] == result_ref:
        return msg.args[1]
    else:
        raise Timeout(f"Timed out after {timeout} epochs")


async def wait_timeout(timeout: int, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
    timeout_ref = Ref()
    send_scheduled(now() + timeout, self(), timeout_ref, hint="timeout")
    result, message = await _wait_inner(
        SelectionMatcher(
            SingleMatcher(args, kwargs),
            _FusedRefSelect((timeout_ref,)),
        )
    )
    if message.args and message.args[0] == timeout_ref:
        _self_node().set_drop_hint(SingleMatcher(args, kwargs))
        raise Timeout(f"Timed out after {timeout} epochs")
    _self_node().set_drop_hint(_FusedRefSelect((timeout_ref,)))
    message = _filter_message(message, result)
    return *message.args, message.kwargs


class Loop(Atom): ...


async def sleep(duration: int):
    ref = Ref()
    send_scheduled(now() + duration, self(), ref, hint="wake")
    await wait(ref)


async def sleep_until(deadline: int):
    ref = Ref()
    send_scheduled(deadline, self(), ref, hint="wake")
    await wait(ref)


class ResponderFn(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Coroutine[Any, Any, Any]: ...


def responder(fn: ResponderFn):
    async def wrapped(sender: Addr, ref: Ref, *args: Any, **kwargs: Any):
        result = await fn(*args, **kwargs)
        send(sender, ref, result)

    return wrapped


def loop(fn: Callable[..., Awaitable[None]]) -> StateMachineInit:
    sm = SMBuilder()

    @sm.handle(Loop)
    async def start():
        send(self(), Loop)
        await fn()

    return initialize(sm, message(Loop))


def launch(fn: Callable[..., Awaitable[None]]) -> StateMachineInit:
    sm = SMBuilder()

    @sm.handle(None)
    async def start():
        await fn()

    return initialize(sm, message(None))
