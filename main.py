import csv
import functools
import hashlib
import inspect
import multiprocessing
import os
import sqlite3
import types
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from random import Random
from typing import (
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Generator,
    Mapping,
    NamedTuple,
    Protocol,
    Self,
    Sequence,
    cast,
)
from uuid import UUID, uuid4

from frozendict import frozendict
from tqdm import tqdm


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
        return f"{type(self).__name__}({str(self.uuid)!r}"


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
    def event_loop(self) -> EventLoop: ...
    @property
    def self_addr(self) -> Addr: ...
    @property
    def sent_messages(self) -> list[tuple[Addr, Message]]: ...


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

    def branch_handler(self, *match_args: Any, **match_kwargs: Any):
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


_builder_var = ContextVar[SMBuilder | None]("builder")


def _builder() -> SMBuilder:
    builder = _builder_var.get()
    if builder is None:
        raise RuntimeError("_builder() called outside of build_state_machine context")
    return builder


@contextmanager
def build_state_machine() -> Generator[StateMachine]:
    builder = SMBuilder()
    token = _builder_var.set(builder)
    try:
        yield builder
    finally:
        _builder_var.reset(token)


def branch(*match_args: Any, **match_kwargs: Any):
    def decorator(fn: HandlerFn[StateMachine]):
        _builder().add_branch(
            matcher=SingleMatcher(match_args, match_kwargs), transition=fn
        )

    return decorator


def async_branch_handler(*match_args: Any, **match_kwargs: Any):
    """Equivalent to the following:

    ```python
    with build_state_machine() as sm:
        @branch(*match_args, **match_kwargs)
        @async_transition(continuation=sm)
        @handler
        async def handle():
            ...
    ```
    """

    def decorator(fn: Callable[..., SMCoroutine]):
        @branch(*match_args, **match_kwargs)
        @async_transition(continuation=_builder())
        @handler
        @functools.wraps(fn)
        def handle(*args: Any, **kwargs: Any) -> SMCoroutine:
            return fn(*args, **kwargs)

    return decorator


def branch_handler(*match_args: Any, **match_kwargs: Any):
    """Equivalent to the following:

    ```python
    with build_state_machine() as sm:
        @branch(*match_args, **match_kwargs)
        @handler
        def handle():
            ...
    ```
    """

    def decorator(fn: Callable[..., StateMachine]):
        branch(*match_args, **match_kwargs)(handler(fn))

    return decorator


with build_state_machine() as sm:

    @branch()
    @handler
    def init():
        return sm

    @branch_handler()
    def init_2():
        return sm

    @branch()
    @async_transition(continuation=sm)
    @handler
    async def start():
        pass

    @async_branch_handler()
    async def handle():
        pass


@dataclass
class ContextImpl:
    event_loop: EventLoop
    self_addr: Addr
    sent_messages: list[tuple[Addr, Message]]


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


class Node:
    def __init__(self, sm: StateMachine):
        self.sm = sm
        self.inbox: list[Message] = []
        self.drop_hints: list[Matcher] = []
        self.fuel_per_epoch: int | None = None
        self.unlimited_fuel = False
        self.stopped = False
        self._detector = _LeakDetector(alpha=0.09, threshold=0.1)
        self._addr = None

    def run(
        self, fuel: int, self_addr: Addr, event_loop: EventLoop
    ) -> list[tuple[Addr, Message]]:
        self._addr = self_addr
        if self.stopped:
            return []
        ctx = ContextImpl(event_loop=event_loop, self_addr=self_addr, sent_messages=[])
        if self.unlimited_fuel:
            range_ = iter(int, 1)
        else:
            range_ = range(fuel)
        for _ in range_:
            result = next_state(self.sm, ctx=ctx, msgs=self.inbox)
            if result is not None:
                i, next_sm = result
                self.sm = next_sm
                self.inbox.pop(i)
            else:
                break
        self._analyze_inbox_growth()
        return ctx.sent_messages

    def _analyze_inbox_growth(self):
        self._detector.update(len(self.inbox))
        if self._detector.is_leaking():
            warnings.warn(f"{self._addr} inbox growing. size={len(self.inbox)}")
            print(self.inbox, self.sm.matcher)

    def deliver(self, msg: Message):
        if self.stopped:
            return
        for i, hint in enumerate(self.drop_hints):
            if hint.match(msg):
                self.drop_hints.pop(i)
                return
        self.inbox.append(msg)

    def set_drop_hint(self, matcher: Matcher):
        for i, msg in enumerate(self.inbox):
            if matcher.match(msg):
                self.inbox.pop(i)
                return
        self.drop_hints.append(matcher)


@dataclass
class DataPoint:
    epoch: int
    data: dict[tuple[Addr, str], Any]


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

    def logs_sqlite(self) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        conn.execute(
            """
            CREATE TABLE log (
                epoch INTEGER,
                addr TEXT,
                msg TEXT,
                data TEXT
            )
            """
        )
        for log in self.logs:
            conn.execute(
                "INSERT INTO log (epoch, addr, msg, data) VALUES (?, ?, ?, ?)",
                (log.epoch, log.addr.name, log.msg, repr(log.data)),
            )
        conn.commit()
        return conn


class StopCondition(Protocol):
    def process_logs(self, logs: list[LogEntry]): ...
    @property
    def should_stop(self) -> bool: ...


class StopAfterNodesExited:
    def __init__(self, nodes: Sequence[Addr], exit_msg: str = "exited"):
        self.nodes = set(nodes)
        self.exit_msg = exit_msg
        self.exited_nodes = set()

    def process_logs(self, logs: list[LogEntry]):
        for log in logs:
            if log.msg == self.exit_msg and log.addr in self.nodes:
                self.exited_nodes.add(log.addr)

    @property
    def should_stop(self) -> bool:
        return self.exited_nodes == self.nodes


class EventLoop:
    def __init__(self, without_defaults: bool = False):
        self.nodes = dict[Addr, Node]()
        self.fuel_per_epoch = 100
        self.logs: list[LogEntry] = []
        self.new_logs: list[LogEntry] = []
        self.epoch = 0
        self.rngs = dict[Addr, Random]()

        if not without_defaults:
            self.spawn_spec(*RUNTIME_SPECS)
            self.nodes[_TIMER_ADDR].unlimited_fuel = True

    def spawn(
        self,
        addr: Addr,
        init: StateMachineInit,
    ):
        node = Node(init.sm)
        node.inbox.extend(init.messages)
        self.nodes[addr] = node
        self.rngs[addr] = Random(hashlib.sha256(addr.name.encode()).digest())

    def spawn_spec(self, *specs: NodeSpec):
        for spec in specs:
            self.spawn(spec.addr, spec.init())

    def run(self, epochs: int = 1, condition: StopCondition | None = None):
        for _ in tqdm(range(epochs), unit="epoch", total=epochs):
            sent_messages: list[tuple[Addr, Message]] = []
            for addr, node in self.nodes.items():
                sent_messages.extend(
                    node.run(
                        fuel=node.fuel_per_epoch or self.fuel_per_epoch,
                        self_addr=addr,
                        event_loop=self,
                    )
                )
            for dest_addr, raw_msg in sent_messages:
                try:
                    self.nodes[dest_addr].deliver(raw_msg)
                except KeyError:
                    raise RuntimeError(f"Unknown address: {dest_addr}")

            if condition is not None:
                condition.process_logs(self.new_logs)
                if condition.should_stop:
                    break

            self.epoch += 1
            self.commit_logs()

        return RunResult(logs=self.logs, num_epochs=self.epoch)

    def commit_logs(self):
        self.logs.extend(self.new_logs)
        self.new_logs.clear()

    def append_log(self, log: LogEntry):
        self.new_logs.append(log)


def send(addr: Addr, *args: Any, **kwargs: Any):
    context().sent_messages.append((addr, message(*args, **kwargs)))


def self() -> Addr:
    return context().self_addr


def self_node() -> Node:
    return context().event_loop.nodes[context().self_addr]


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


async def select(*options: Message) -> tuple[Any, ...]:
    _, msg = await _wait_inner(
        SelectionMatcher(*(SingleMatcher(opt.args, opt.kwargs) for opt in options))
    )
    return *msg.args, msg.kwargs


def log(msg: str | None = None, /, **kwargs: Any):
    if msg is not None:
        kwargs["_msg"] = msg
    context().event_loop.append_log(LogEntry(epoch=now(), addr=self(), data=kwargs))


def now() -> int:
    return context().event_loop.epoch


def rng() -> Random:
    return context().event_loop.rngs[self()]


async def ask(addr: Addr, method: Any, *args: Any, **kwargs: Any) -> Any:
    ref = Ref()
    send(addr, method, self(), ref, *args, **kwargs)
    result, *_ = await wait(ref)
    return result


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
    deadline: int | None = None,
) -> Any:
    result_ref = Ref()
    send(addr, method, self(), result_ref, *args)
    sleep_ref = Ref()
    send(_TIMER_ADDR, Sleep, self(), sleep_ref, timeout)
    deadline_ref = Ref()
    if deadline is not None:
        send(_TIMER_ADDR, SleepUntil, self(), sleep_ref, deadline)
    result, msg = await _wait_inner(
        _FusedRefSelect((result_ref, sleep_ref, deadline_ref))
    )
    if result.branch_id == 0:
        self_node().set_drop_hint(SingleMatcher((sleep_ref,), {}))
        self_node().set_drop_hint(SingleMatcher((deadline_ref,), {}))
        return msg.args[1]
    else:
        self_node().set_drop_hint(SingleMatcher((result_ref,), {}))
        if result.branch_id == 1:
            self_node().set_drop_hint(SingleMatcher((deadline_ref,), {}))
            raise Timeout(f"Timed out after {timeout} epochs")
        else:
            self_node().set_drop_hint(SingleMatcher((sleep_ref,), {}))
            raise Timeout(f"Timed out at deadline epoch {deadline}", is_deadline=True)


class Loop(Atom): ...


class Sleep(Atom): ...


class SleepUntil(Atom): ...


_TIMER_ADDR = Addr("__internal.timer")


def sleep(duration: int):
    return ask(_TIMER_ADDR, Sleep, duration)


def sleep_until(deadline: int):
    return ask(_TIMER_ADDR, SleepUntil, deadline)


class NodeSpec(Protocol):
    @property
    def addr(self) -> Addr: ...
    def init(self) -> StateMachineInit: ...


class TimerSpec:
    addr = _TIMER_ADDR

    def init(_self):
        with build_state_machine() as timer:
            waiting_tasks: dict[Ref, tuple[Addr, int]] = {}

            @async_branch_handler(Sleep)
            async def sleep(sender: Addr, ref: Ref, duration: int):
                wake_time = now() + duration
                waiting_tasks[ref] = (sender, wake_time)

            @async_branch_handler(SleepUntil)
            async def sleep_until(sender: Addr, ref: Ref, until: int):
                if until <= now():
                    send(sender, ref)
                    return
                wake_time = until
                waiting_tasks[ref] = (sender, wake_time)

            @async_branch_handler(Loop)
            async def loop():
                send(self(), Loop)

                current_time = now()
                to_wake = [
                    ref
                    for ref, (_, wake_time) in waiting_tasks.items()
                    if wake_time <= current_time
                ]
                for ref in to_wake:
                    sender, _ = waiting_tasks.pop(ref)
                    send(sender, ref, hint="wake")

        return initialize(timer, message(Loop))


RUNTIME_SPECS = (TimerSpec(),)


class ResponderFn(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Coroutine[Any, Any, Any]: ...


def responder(fn: ResponderFn):
    async def wrapped(sender: Addr, ref: Ref, *args: Any, **kwargs: Any):
        result = await fn(*args, **kwargs)
        send(sender, ref, result)

    return wrapped


def loop(fn: Callable[..., Awaitable[None]]) -> StateMachineInit:
    with build_state_machine() as sm:

        @async_branch_handler(Loop)
        async def start():
            send(self(), Loop)
            await fn()

    return initialize(sm, message(Loop))


def launch(fn: Callable[..., Awaitable[None]]) -> StateMachineInit:
    with build_state_machine() as t:

        @async_branch_handler(None)
        async def start():
            await fn()

    return initialize(t, message(None))


def test_simple():
    class Add(Atom): ...

    def make_adder():
        with build_state_machine() as adder:

            @async_branch_handler(Add)
            async def add(sender: Addr, ref: Ref, a: int, b: int):
                send(sender, ref, a + b)

        return adder

    def make_main(adder: Addr):
        @dataclass
        class State:
            i: int = 0

        state = State()

        @loop
        async def main():
            result = await ask(adder, Add, state.i, 10)
            log(result=result)

            state.i += 1

        return main

    adder_addr = Addr("adder")

    event_loop = EventLoop()
    event_loop.spawn(adder_addr, make_adder())
    event_loop.spawn(Addr("main"), make_main(adder_addr))
    event_loop.run(epochs=50)

    for entry in event_loop.logs:
        print(entry)

    values = [entry.data["result"] for entry in event_loop.logs]
    for i in range(10, 20):
        assert i in values


class Enqueue(Atom): ...


class Dequeue(Atom): ...


class QueueFull(Atom): ...


class Ok(Atom): ...


class Err(Atom): ...


def make_queue(max_size: int = 10) -> StateMachineInit:
    items = []
    waiting: list[tuple[Addr, Ref]] = []

    queue = SMBuilder()

    def log_size():
        log("queue size", size=len(items))

    @queue.branch_handler(Enqueue)
    async def enqueue(sender: Addr, ref: Ref, item: Any):
        if len(items) >= max_size:
            send(sender, ref, QueueFull)
        else:
            items.append(item)
            while waiting and items:
                send(*waiting.pop(0), items.pop(0))
            log_size()
            send(sender, ref, Ok, hint="enqueued")

    @queue.branch_handler(Dequeue)
    async def dequeue(sender: Addr, ref: Ref):
        if items:
            send(sender, ref, items.pop(0))
            log_size()
        else:
            waiting.append((sender, ref))

    return queue


@dataclass
class Queue[T]:
    addr: Addr

    async def enqueue(self, item: T) -> bool:
        result = await ask(self.addr, Enqueue, item)
        return result is not QueueFull

    async def dequeue(self) -> T:
        return await ask(self.addr, Dequeue)


def test_producer_consumer():
    def make_consumer(queue: Queue[int]):
        @launch
        async def start():
            while True:
                item = await queue.dequeue()
                log(dequeued=item)

        return start

    def make_producer(queue: Queue[int]):
        @launch
        async def start():
            await sleep(20)
            for i in range(20):
                await queue.enqueue(i)
                log(enqueued=i)

        return start

    queue_addr = Addr("queue")

    event_loop = EventLoop()
    event_loop.spawn(queue_addr, make_queue(max_size=30))
    event_loop.spawn(Addr("producer"), make_producer(Queue(queue_addr)))
    event_loop.spawn(Addr("consumer"), make_consumer(Queue(queue_addr)))

    event_loop.run(epochs=100)

    for entry in event_loop.logs:
        print(entry)

    dequeued = set(
        entry.data["dequeued"] for entry in event_loop.logs if "dequeued" in entry.data
    )
    assert dequeued == set(range(20))


def test_timeout():
    @loop
    async def do_nothing():
        await sleep(200)

    @launch
    async def main():
        try:
            await ask_timeout(10, Addr("do_nothing"), None)
        except TimeoutError:
            log("timed out")

    event_loop = EventLoop()
    event_loop.spawn(Addr("do_nothing"), do_nothing)
    event_loop.spawn(Addr("main"), main)
    event_loop.run(epochs=50)

    assert any(
        log.msg == "timed out" for log in event_loop.logs if log.addr.name == "main"
    )


def constant_retry(sleep_time: int):
    def policy(_try_num: int) -> int:
        return sleep_time

    return policy


def run_experiment(
    num_clients: int = 10,
    num_clients_spike: int = 50,
    num_clients_after_spike: int = 30,
    num_workers: int = 4,
    queue_size: int = 14,
    num_epochs: int = 30000,
    num_retries: int = 3,
    retry_policy: Callable[[int], int] = constant_retry(20),
    spike_offset: int = 10000,
    spike_duration: int = 3000,
    submit_timeout: int = 30,
    work_time_range: tuple[int, int] = (5, 20),
    inter_task_sleep_range: tuple[int, int] = (10, 30),
):
    def make_worker(queue: Queue[tuple[Addr, Ref]]):
        @loop
        async def start():
            addr, ref = await queue.dequeue()
            log("start task", ref=ref)
            await sleep(rng().randrange(*work_time_range))
            log("finish task", ref=ref)
            send(addr, ref, Ok, hint="task_finished")

        return start

    queue_addrs = [Addr(f"queue-{i}") for i in range(num_workers)]

    load_balancer = SMBuilder()

    class SubmitWork(Atom): ...

    class Outstanding(NamedTuple):
        sender: Addr
        ref: Ref
        queue: Addr

    lb_outstanding: dict[Ref, Outstanding] = {}

    @load_balancer.branch_handler(SubmitWork)
    async def submit_work(sender: Addr, ref: Ref):
        worker_queue_addr = rng().choice(queue_addrs)
        send(worker_queue_addr, Enqueue, self(), ref, (sender, ref))
        lb_outstanding[ref] = Outstanding(
            sender=sender, ref=ref, queue=worker_queue_addr
        )

    @load_balancer.branch_handler(..., QueueFull)
    async def handle_queue_full(ref: Ref):
        outstanding = lb_outstanding.pop(ref, None)
        assert outstanding is not None, "Received QueueFull for unknown submission"
        log("queue full", queue=outstanding.queue)
        send(outstanding.sender, outstanding.ref, Err, hint="work_failed")

    @load_balancer.branch_handler(..., Ok)
    async def handle_ok(ref: Ref):
        outstanding = lb_outstanding.pop(ref, None)
        assert outstanding is not None, "Received Ok for unknown submission"

    lb_addr = Addr("load_balancer")

    async def submit_work_wrapper(
        n_retries: int = 3,
        timeout: int | None = None,
        policy: Callable[[int], int] = constant_retry(10),
        deadline: int | None = None,
    ) -> bool:
        submission_id = Ref()
        for try_num in range(n_retries):
            if deadline is not None and now() >= deadline:
                log(
                    "deadline exceeded",
                    deadline=deadline,
                    try_num=try_num,
                    submission_id=submission_id,
                )
                return False
            if timeout:
                try:
                    result = await ask_timeout(
                        timeout,
                        lb_addr,
                        SubmitWork,
                    )
                except Timeout:
                    log(
                        "timeout",
                        after=timeout,
                        try_num=try_num,
                        submission_id=submission_id,
                    )
                    result = Err()
            else:
                result = await ask(lb_addr, SubmitWork)
            match result:
                case Ok():
                    return True
                case Err():
                    await sleep(policy(try_num))
                    log("retry", try_num=try_num, submission_id=submission_id)
                case _:
                    assert False
        return False

    async def perform_work(deadline: int | None = None):
        start = now()
        success = await submit_work_wrapper(
            n_retries=num_retries,
            timeout=submit_timeout,
            policy=retry_policy,
            deadline=deadline,
        )
        if success:
            log("finished", latency=now() - start, start=start)
        else:
            log("failed", latency=now() - start)

    clients: list[StateMachineInit] = []

    def make_normal_client(client_ix: int):
        @launch
        async def client():
            log("client started", client_ix=client_ix)
            while True:
                await perform_work()
                await sleep(rng().randrange(*inter_task_sleep_range))

        return client

    def make_spike_client(client_ix: int):
        @launch
        async def client():
            await sleep(spike_offset + rng().randrange(0, 300))
            log("client started", client_ix=client_ix)
            while True:
                await perform_work(
                    deadline=spike_offset + spike_duration + rng().randrange(0, 300)
                )
                if now() >= spike_offset + spike_duration:
                    break
                await sleep(rng().randrange(*inter_task_sleep_range))
            log("client exited", client_ix=client_ix)
            self_node().stopped = True

        return client

    def make_added_client(client_ix: int):
        @launch
        async def client():
            await sleep(spike_offset + rng().randrange(0, 300))
            log("client started", client_ix=client_ix)
            while True:
                await perform_work()
                await sleep(rng().randrange(*inter_task_sleep_range))

        return client

    for client_ix in range(
        max(num_clients, num_clients_spike, num_clients_after_spike)
    ):
        if client_ix < num_clients:
            clients.append(make_normal_client(client_ix))
        elif client_ix < num_clients_after_spike:
            clients.append(make_added_client(client_ix))
        elif client_ix < num_clients_spike:
            clients.append(make_spike_client(client_ix))

    event_loop = EventLoop()
    for i, queue_addr in enumerate(queue_addrs):
        event_loop.spawn(queue_addr, make_queue(max_size=queue_size))
        event_loop.spawn(Addr(f"worker-{i}"), make_worker(Queue(queue_addr)))
    event_loop.spawn(lb_addr, load_balancer)
    for i, client in enumerate(clients):
        event_loop.spawn(Addr(f"client-{i}"), client)

    result = event_loop.run(epochs=num_epochs)
    return result


def analyze_result(result: RunResult):
    import numpy as np
    import polars as pl

    with result.logs_sqlite() as conn:
        client_starts = [
            t
            for (t,) in conn.execute(
                "select epoch from log where msg='client started'"
            ).fetchall()
        ]
        client_stops = [
            t
            for (t,) in conn.execute(
                "select epoch from log where msg='client exited'"
            ).fetchall()
        ]
        start_counts = np.bincount(client_starts, minlength=result.num_epochs)
        stop_counts = np.bincount(client_stops, minlength=result.num_epochs)
        num_clients = np.cumsum(start_counts - stop_counts)

        task_latency = [
            (t, lt)
            for t, lt in conn.execute(
                "select data->>'start', data->>'latency' from log where msg='finished'"
            ).fetchall()
        ]
        latency_df = pl.DataFrame(task_latency, schema=["time", "value"], orient="row")
        averages = latency_df.group_by("time").agg(pl.col("value").mean())
        latency_result_df = (
            pl.DataFrame({"time": range(result.num_epochs)})
            .join(averages, on="time", how="left")
            .with_columns(pl.col("value").forward_fill().fill_null(0))
        )

        smoothing_window = 300
        smoothed_latency = np.convolve(
            latency_result_df["value"].to_numpy(),
            np.ones(smoothing_window) / smoothing_window,
            mode="same",
        )

        queue_size = [
            (t, sz)
            for t, sz in conn.execute(
                "select epoch, data->>'size' from log where msg='queue size'"
            ).fetchall()
        ]
        queue_size_df = pl.DataFrame(queue_size, schema=["time", "value"], orient="row")
        averages = queue_size_df.group_by("time").agg(pl.col("value").mean())
        queue_size_result_df = (
            pl.DataFrame({"time": range(result.num_epochs)})
            .join(averages, on="time", how="left")
            .with_columns(pl.col("value").forward_fill().fill_null(0))
        )
        smoothed_queue_size = np.convolve(
            queue_size_result_df["value"].to_numpy(),
            np.ones(smoothing_window) / smoothing_window,
            mode="same",
        )

        successes = [
            t
            for (t,) in conn.execute(
                "select epoch from log where msg='finished'"
            ).fetchall()
        ]
        failures = [
            t
            for (t,) in conn.execute(
                "select epoch from log where msg='failed'"
            ).fetchall()
        ]
        success_counts = np.bincount(successes, minlength=result.num_epochs)
        failure_counts = np.bincount(failures, minlength=result.num_epochs)
        total_successes = np.cumsum(success_counts)
        total_failures = np.cumsum(failure_counts)

        df = pl.DataFrame(
            {
                "time": range(result.num_epochs),
                "num_clients": num_clients,
                "task_latency": latency_result_df["value"],
                "smoothed_task_latency": smoothed_latency,
                "queue_size": queue_size_result_df["value"],
                "smoothed_queue_size": smoothed_queue_size,
                "total_successes": total_successes,
                "total_failures": total_failures,
            }
        )
        return df


def write_csv(path: str, data: list[dict]):
    with open(path, newline="", mode="w") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)


class Runner:
    def __init__(self):
        self.runs = []

    def trial(self, values: Sequence[Any], name: str | None = None):
        def decorator(fn: Callable[..., Any]):
            trial_name = name or fn.__name__.replace("_", "-")
            self.runs.append((trial_name, fn, values))
            return fn

        return decorator

    def _runner_fn(self, args: tuple[str, Callable[..., Any], Any, str]):
        import polars as pl

        _name, fn, input, output_path = args
        result = fn(input)
        if isinstance(result, pl.DataFrame):
            result.write_csv(output_path, include_header=True)
        elif isinstance(result, list):
            write_csv(output_path, result)
        else:
            raise TypeError("Unsupported result type")

    def run_all(self, force: bool = False, concurrent: bool = True):
        runs = []
        for name, fn, inputs in self.runs:
            for input in inputs:
                output_path = f"results/{name}-{input}.csv"
                if os.path.exists(output_path) and not force:
                    print(f"Skip {name} with input {input} (already exists)")
                    continue
                runs.append((name, fn, input, output_path))

        print("To run")
        for name, *_ in runs:
            print(f"  {name}")

        if concurrent:
            pool = multiprocessing.Pool()
            list(tqdm(pool.imap_unordered(self._runner_fn, runs), total=len(runs)))
        else:
            for run in tqdm(runs):
                self._runner_fn(run)


runner = Runner()


@runner.trial([1])
def client_load_spike(_):
    return analyze_result(
        # 1/20 tps
        # avg task latency = 6 * 20 = 120
        # single req rate = 1/(120 + 5)
        run_experiment(
            num_workers=1,
            queue_size=10,
            num_retries=100_000,
            num_epochs=24_000,
            spike_offset=8000,
            spike_duration=6000,
            num_clients=3,
            num_clients_spike=5,
            num_clients_after_spike=3,
            submit_timeout=120,
            work_time_range=(33, 37),
            inter_task_sleep_range=(28, 32),
            retry_policy=constant_retry(0),
        )
    )


def main():
    runner.run_all(force=True, concurrent=False)


if __name__ == "__main__":
    raise SystemExit(main())
