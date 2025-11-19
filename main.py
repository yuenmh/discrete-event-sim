import functools
import hashlib
import inspect
import types
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
    Protocol,
    Self,
    Sequence,
    cast,
)
from uuid import UUID, uuid4

from frozendict import frozendict


@dataclass(frozen=True, slots=True)
class Message:
    args: tuple[Any, ...]
    kwargs: frozendict[str, Any]


def message(*args: Any, **kwargs: Any) -> Message:
    return Message(args, frozendict(kwargs))


@dataclass(frozen=True, slots=True)
class Addr:
    name: str


@dataclass(frozen=True, slots=True)
class Ref:
    uuid: UUID = field(default_factory=uuid4)


class Atom:
    pass


class Ctx:
    def __init__(
        self,
        state: dict[str, Any],
        self_addr: Addr,
        epoch_idx: int,
        event_loop: EventLoop,
    ) -> None:
        self.state = state
        self.self = self_addr
        self.sent_messages: list[tuple[Addr, Message]] = []
        self.epoch_idx = epoch_idx
        self._event_loop = event_loop

    def send(self, addr: Addr, *args: Any, **kwargs: Any) -> None:
        self.sent_messages.append((addr, Message(args, frozendict(kwargs))))

    @types.coroutine
    def wait(
        self, *args: Any, **kwargs: Any
    ) -> Generator[CoBranchYield, Message, tuple[Any, ...]]:
        msg = yield CoBranchYield(args, kwargs)
        return *msg.args, msg.kwargs


class TransitionFn(Protocol):
    def __call__(self, ctx: Ctx, msg: Message) -> "Transition": ...


class TransitionImplFn(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> "Transition": ...


def _wrap_normal_transition_fn(fn: TransitionImplFn) -> TransitionFn:
    sig = inspect.signature(fn)

    def wrapped(ctx: Ctx, msg: Message) -> Transition:
        bound = sig.bind(*msg.args, **msg.kwargs)
        with scoped_ctx(ctx):
            return fn(*bound.args, **bound.kwargs)

    return wrapped


@dataclass
class CoBranchYield:
    match_args: tuple[Any, ...]
    match_kwargs: dict[str, Any]


class TransitionCoroutineFn(Protocol):
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> Coroutine[CoBranchYield, Message | None, None]: ...


_ctx = ContextVar[Ctx | None]("ctx")


@contextmanager
def scoped_ctx(ctx: Ctx):
    token = _ctx.set(ctx)
    try:
        yield
    finally:
        _ctx.reset(token)


def ctx() -> Ctx:
    cx = _ctx.get()
    if cx is None:
        raise RuntimeError("ctx() called outside of a scoped context")
    return cx


def _wrap_coroutine_transition_fn(
    fn: TransitionCoroutineFn, parent_transition: Transition
) -> TransitionFn:
    sig = inspect.signature(fn)

    def wrapped(ctx: Ctx, msg: Message) -> Transition:
        bound = sig.bind(*msg.args, **msg.kwargs)
        with scoped_ctx(ctx):
            co = fn(*bound.args, **bound.kwargs)
            try:
                yield_params = co.send(None)
            except StopIteration:
                return parent_transition
        t = Transition()

        @t.add_branch(*yield_params.match_args, **yield_params.match_kwargs)
        def resume(ctx: Ctx, msg: Message):
            try:
                with scoped_ctx(ctx):
                    yield_params = co.send(msg)
            except StopIteration:
                return parent_transition
            t.branches[0].match_args = yield_params.match_args
            t.branches[0].match_kwargs = yield_params.match_kwargs
            return t

        return t

    return wrapped


class Transition:
    def __init__(
        self,
        branches: list[TransitionBranch] | None = None,
        init_messages: Sequence[Message] = (),
    ) -> None:
        self.branches = branches or []
        self.init_messages = init_messages

    def add_branch(self, *args: Any, **kwargs: Any):
        def decorator(fn: TransitionFn):
            self.branches.append(
                TransitionBranch(
                    match_args=args,
                    match_kwargs=kwargs,
                    fn=fn,
                )
            )

        return decorator

    def branch(self, *args: Any, **kwargs: Any):
        def decorator(fn: TransitionImplFn | TransitionCoroutineFn):
            if inspect.iscoroutinefunction(fn):
                self.add_branch(*args, **kwargs)(
                    _wrap_coroutine_transition_fn(fn, self)
                )
            elif inspect.isfunction(fn):
                self.add_branch(*args, **kwargs)(_wrap_normal_transition_fn(fn))

        return decorator

    def find_matching_branch(self, msg: Message) -> TransitionBranch | None:
        for branch in self.branches:
            if branch.matches(msg):
                return branch
        return None

    def find_message(
        self, msgs: list[Message]
    ) -> tuple[int, Message, TransitionBranch] | None:
        for i, msg in enumerate(msgs):
            branch = self.find_matching_branch(msg)
            if branch is not None:
                msg = replace(msg, args=msg.args[len(branch.match_args) :])
                return i, msg, branch
        return None


class TransitionBranch:
    def __init__(
        self,
        match_args: tuple[Any, ...],
        match_kwargs: dict[str, Any],
        fn: TransitionFn,
    ) -> None:
        self.match_args = match_args
        self.match_kwargs = match_kwargs
        self.fn = fn

    def matches(self, msg: Message) -> bool:
        if (
            len(msg.args) >= len(self.match_args)
            and msg.args[: len(self.match_args)] == self.match_args
        ):
            for key, value in self.match_kwargs.items():
                if key not in msg.kwargs or msg.kwargs[key] != value:
                    return False
            return True
        return False

    def call(self, ctx: Ctx, msg: Message) -> Transition:
        return self.fn(ctx, msg)


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


class SelectionMatcher(Matcher):
    def __init__(self, *options: Matcher):
        self.options = options

    def match(self, msg: Message) -> MatchResult:
        for option in self.options:
            result = option.match(msg)
            if result.matched:
                return result
        return MatchResult(matched=False)


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


def next_state(
    sm: StateMachine, ctx: Context, msgs: Sequence[Message]
) -> tuple[int, StateMachine] | None:
    for i, msg in enumerate(msgs):
        if match := sm.matcher.match(msg):
            next_sm = sm.next(ctx, match, msg)
            return i, next_sm
    return None


@dataclass
class StateMachineInit:
    sm: StateMachine
    messages: Sequence[Message] = ()


def initialize(sm: StateMachine, *msgs: Message) -> StateMachineInit:
    return StateMachineInit(sm, msgs)


@dataclass
class ResumeSM:
    ctx: Context
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


class CoroutineSM:
    def __init__(self, co: SMCoroutine, continuation: StateMachine | None = None):
        self.co = co
        try:
            self.matcher = co.send(None)
        except StopIteration:
            self.matcher = SelectionMatcher()
        self.continuation = continuation

    def next(self, ctx: Context, match: MatchResult, msg: Message) -> StateMachine:
        try:
            next_matcher = self.co.send(ResumeSM(ctx=ctx, match=match, msg=msg))
            self.matcher = next_matcher
            return self
        except StopIteration:
            if self.continuation is not None:
                return self.continuation

        self.matcher = SelectionMatcher()
        return self


class SMBuilderTransition(Protocol):
    def __call__(self, ctx: Context, msg: Message) -> StateMachine: ...


class SMBuilder:
    def __init__(self):
        self.root_sm = BuilderSM()
        self.branches: list[tuple[Matcher, SMBuilderTransition]] = []

    def add_branch(self, matcher: Matcher, transition: SMBuilderTransition):
        self.branches.append((matcher, transition))

    def build(self) -> StateMachine:
        for i, (_, transition) in enumerate(self.branches):
            self.root_sm.branches[i] = transition
        matcher = SelectionMatcher(
            *(
                BranchIDMatcher(matcher, branch_id=i)
                for i, (matcher, _) in enumerate(self.branches)
            )
        )
        self.root_sm.matcher = matcher
        return self.root_sm


class BuilderSM:
    def __init__(self):
        self.matcher = SelectionMatcher()
        self.branches = dict[int, SMBuilderTransition]()

    def next(self, ctx: Context, match: MatchResult, msg: Message) -> StateMachine:
        transition = self.branches.get(match.branch_id)
        assert transition is not None, "No transition for branch_id"
        return transition(ctx, _filter_message(msg, match))


class HandlerFn[T](Protocol):
    def __call__(self, ctx: Context, msg: Message) -> T: ...


_context_var = ContextVar[Context | None]("context")


def handler[T](fn: Callable[..., T]) -> HandlerFn[T]:
    sig = inspect.signature(fn)

    def wrapped(ctx: Context, msg: Message) -> T:
        bound = sig.bind(*msg.args, **msg.kwargs)
        token = _context_var.set(ctx)
        result = fn(*bound.args, **bound.kwargs)
        _context_var.reset(token)
        return result

    return wrapped


class SMBuilderAsyncTransition(Protocol):
    def __call__(self, ctx: Context, msg: Message) -> SMCoroutine: ...


def async_transition(continuation: StateMachine | None = None):
    def decorator(transition: SMBuilderAsyncTransition):
        @functools.wraps(transition)
        def wrapped(ctx: Context, msg: Message) -> StateMachine:
            return CoroutineSM(co=transition(ctx, msg), continuation=continuation)

        return wrapped

    return decorator


_builder_var = ContextVar[SMBuilder | None]("builder")


def _builder() -> SMBuilder:
    builder = _builder_var.get()
    if builder is None:
        raise RuntimeError("Called outside of build_state_machine context")
    return builder


@contextmanager
def build_state_machine() -> Generator[StateMachine]:
    builder = SMBuilder()
    token = _builder_var.set(builder)
    try:
        yield builder.root_sm
    finally:
        _builder_var.reset(token)


def branch(*match_args: Any, **match_kwargs: Any):
    def decorator(fn: HandlerFn[StateMachine]):
        _builder().add_branch(
            matcher=SingleMatcher(match_args, match_kwargs),
            transition=cast(HandlerFn[StateMachine], fn),
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
        branch(*match_args, **match_kwargs)(
            async_transition(continuation=_builder().root_sm)(handler(fn))
        )

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


class Node:
    def __init__(self, sm: StateMachine):
        self.sm = sm
        self.inbox: list[Message] = []

    def run(
        self, fuel: int, self_addr: Addr, event_loop: EventLoop
    ) -> list[tuple[Addr, Message]]:
        ctx = ContextImpl(event_loop=event_loop, self_addr=self_addr, sent_messages=[])
        for _ in range(fuel):
            result = next_state(self.sm, ctx=ctx, msgs=self.inbox)
            if result is not None:
                i, next_sm = result
                self.sm = next_sm
                self.inbox.pop(i)
            else:
                break
        return ctx.sent_messages


@dataclass
class DataPoint:
    epoch: int
    data: dict[tuple[Addr, str], Any]


@dataclass(frozen=True, slots=True)
class LogEntry:
    epoch: int
    addr: Addr
    data: dict[str, Any]

    def __str__(self) -> str:
        if "_msg" in self.data:
            msg = f"{self.data['_msg']} "
        else:
            msg = ""
        data = " ".join(f"{k}={v!r}" for k, v in self.data.items() if k != "_msg")
        return f"[{self.epoch:06}] [{self.addr.name}] {msg}{data}"


class EventLoop:
    def __init__(self):
        self.nodes = dict[Addr, Node]()
        self.fuel_per_epoch = 10
        self.data_collection_paths: list[tuple[Addr, str]] = []
        self.data: list[DataPoint] = []
        self.logs: list[LogEntry] = []
        self.epoch = 0
        self.rngs = dict[Addr, Random]()

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

    def run(self, epochs: int = 1):
        for _ in range(epochs):
            sent_messages: list[tuple[Addr, Message]] = []
            for addr, node in self.nodes.items():
                sent_messages.extend(
                    node.run(
                        fuel=self.fuel_per_epoch,
                        self_addr=addr,
                        event_loop=self,
                    )
                )
            for dest_addr, raw_msg in sent_messages:
                try:
                    self.nodes[dest_addr].inbox.append(raw_msg)
                except KeyError:
                    raise RuntimeError(f"Unknown address: {dest_addr}")
            self.epoch += 1


def send(addr: Addr, *args: Any, **kwargs: Any):
    ctx().send(addr, *args, **kwargs)


def self() -> Addr:
    return ctx().self


async def wait(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
    return await ctx().wait(*args, **kwargs)


def log(msg: str | None = None, /, **kwargs: Any):
    if msg is not None:
        kwargs["_msg"] = msg
    ctx()._event_loop.logs.append(
        LogEntry(epoch=ctx().epoch_idx, addr=ctx().self, data=kwargs)
    )


async def ask(addr: Addr, method: Any, *args: Any, **kwargs: Any) -> Any:
    ref = Ref()
    send(addr, method, self(), ref, *args, **kwargs)
    result, *_ = await wait(ref)
    return result


def now() -> int:
    return ctx().epoch_idx


def rng() -> Random:
    return ctx()._event_loop.rngs[ctx().self]


class Loop(Atom): ...


class Sleep(Atom): ...


_TIMER_ADDR = Addr("__internal.timer")


def sleep(duration: int):
    return ask(_TIMER_ADDR, Sleep, duration)


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
                    send(sender, ref)

        return StateMachineInit(timer, [message(Loop)])


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
            @responder
            async def add(a: int, b: int):
                return a + b

        return initialize(adder)

    def make_main(adder: Addr):
        @dataclass
        class State:
            i: int = 0

        state = State()

        with build_state_machine() as main:

            @async_branch_handler(Loop)
            async def start():
                send(self(), Loop)

                result = await ask(adder, Add, state.i, 10)
                log(result=result)

                await sleep(1)

                state.i += 1

        return initialize(main, message(Loop))

    adder_addr = Addr("adder")

    event_loop = EventLoop()
    event_loop.spawn_spec(*RUNTIME_SPECS)
    event_loop.spawn(adder_addr, make_adder())
    event_loop.spawn(Addr("main"), make_main(adder_addr))
    event_loop.run(epochs=100)

    for entry in event_loop.logs:
        print(entry)

    values = [entry.data["result"] for entry in event_loop.logs]
    for i in range(10, 20):
        assert i in values


class Enqueue(Atom): ...


class Dequeue(Atom): ...


class QueueFull(Atom): ...


class Ok(Atom): ...


def make_queue(max_size: int = 10):
    items = []
    waiting: list[tuple[Addr, Ref]] = []

    queue = Transition()

    def log_size():
        log(size=len(items))

    @queue.branch(Enqueue)
    async def enqueue(sender: Addr, ref: Ref, item: Any):
        if len(items) >= max_size:
            send(sender, ref, QueueFull)
        else:
            items.append(item)
            while waiting and items:
                send(*waiting.pop(0), items.pop(0))
            log_size()
            send(sender, ref, Ok)

    @queue.branch(Dequeue)
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
    event_loop.spawn_spec(*RUNTIME_SPECS)
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


def main():
    def make_worker(queue: Queue[None]):
        @loop
        async def start():
            await queue.dequeue()
            log("start task")
            await sleep(rng().randrange(5, 20))
            log("finish task")

        return start


if __name__ == "__main__":
    raise SystemExit(main())
