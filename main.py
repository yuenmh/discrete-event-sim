import csv
import functools
import hashlib
import inspect
import multiprocessing
import sqlite3
import types
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from operator import itemgetter
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


@dataclass(frozen=True, slots=True)
class Ref:
    uuid: UUID = field(default_factory=_deterministic_uuid)

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
        bound = sig.bind(*msg.args, **msg.kwargs)
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


class Node:
    def __init__(self, sm: StateMachine):
        self.sm = sm
        self.inbox: list[Message] = []
        self.drop_hints: list[Matcher] = []
        self.fuel_per_epoch: int | None = None
        self.unlimited_fuel = False

    def run(
        self, fuel: int, self_addr: Addr, event_loop: EventLoop
    ) -> list[tuple[Addr, Message]]:
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
        return ctx.sent_messages

    def deliver(self, msg: Message):
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
        self.fuel_per_epoch = 10
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
        for _ in range(epochs):
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


async def ask_timeout(
    timeout: int, addr: Addr, method: Any, *args: Any, **kwargs: Any
) -> Any:
    result_ref = Ref()
    send(addr, method, self(), result_ref, *args, **kwargs)
    sleep_ref = Ref()
    send(_TIMER_ADDR, Sleep, self(), sleep_ref, timeout)
    result, msg = await _wait_inner(_FusedRefSelect((result_ref, sleep_ref)))
    if result.branch_id == 0:
        self_node().set_drop_hint(SingleMatcher((sleep_ref,), {}))
        return msg.args[1]
    else:
        self_node().set_drop_hint(SingleMatcher((result_ref,), {}))
        raise TimeoutError(f"Timed out after {timeout} epochs")


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
        log(size=len(items))

    @queue.branch_handler(Enqueue)
    async def enqueue(sender: Addr, ref: Ref, item: Any):
        if len(items) >= max_size:
            send(sender, ref, QueueFull)
        else:
            items.append(item)
            while waiting and items:
                send(*waiting.pop(0), items.pop(0))
            log_size()
            send(sender, ref, Ok)

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


def run_experiment(
    num_clients: int = 10,
    num_workers: int = 4,
    queue_size: int = 14,
    num_epochs: int = 20000,
):
    def make_worker(queue: Queue[tuple[Addr, Ref]]):
        @loop
        async def start():
            addr, ref = await queue.dequeue()
            log("start task", ref=ref)
            await sleep(rng().randrange(5, 20))
            log("finish task", ref=ref)
            send(addr, ref, Ok)

        return start

    queue_addrs = [Addr(f"queue-{i}") for i in range(num_workers)]

    load_balancer = SMBuilder()

    class SubmitWork(Atom): ...

    @load_balancer.branch_handler(SubmitWork)
    async def submit_work(sender: Addr, ref: Ref):
        worker_queue_addr = rng().choice(queue_addrs)
        if await ask(worker_queue_addr, Enqueue, (sender, ref)) is QueueFull:
            log("queue full", queue=worker_queue_addr)
            send(sender, ref, Err)

    lb_addr = Addr("load_balancer")

    def constant_retry(sleep_time: int):
        def policy(_try_num: int) -> int:
            return sleep_time

        return policy

    async def submit_work_wrapper(
        n_retries: int = 5,
        timeout: int | None = None,
        policy: Callable[[int], int] = constant_retry(10),
    ) -> bool:
        submission_id = Ref()
        for try_num in range(n_retries):
            if timeout:
                try:
                    result = await ask_timeout(timeout, lb_addr, SubmitWork)
                except TimeoutError:
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

    async def perform_work():
        start = now()
        success = await submit_work_wrapper(
            n_retries=3, timeout=100, policy=constant_retry(20)
        )
        if success:
            log("finished", latency=now() - start)
        else:
            log("failed", latency=now() - start)

    clients: list[StateMachineInit] = []

    for _ in range(num_clients):

        @launch
        async def client():
            for _ in range(50):
                await perform_work()
            log("exited")

        clients.append(client)

    event_loop = EventLoop()
    for i, queue_addr in enumerate(queue_addrs):
        event_loop.spawn(queue_addr, make_queue(max_size=queue_size))
        event_loop.spawn(Addr(f"worker-{i}"), make_worker(Queue(queue_addr)))
    event_loop.spawn(lb_addr, load_balancer)
    for i, client in enumerate(clients):
        event_loop.spawn(Addr(f"client-{i}"), client)

    condition = StopAfterNodesExited([Addr(f"client-{i}") for i in range(num_clients)])
    result = event_loop.run(epochs=num_epochs, condition=condition)
    return result


def analyze_result(result: RunResult):
    with result.logs_sqlite() as conn:
        size = [
            x
            for (x,) in conn.execute(
                "select data->>'size' from log where addr like 'queue-%'"
            ).fetchall()
        ]
        latency = [
            x
            for (x,) in conn.execute(
                "select data->>'latency' from log where addr like 'client-%' and msg='finished'"
            ).fetchall()
        ]
        failed_latency = [
            x
            for (x,) in conn.execute(
                "select data->>'latency' from log where addr like 'client-%' and msg='failed'"
            ).fetchall()
        ]

        datum = {
            "num_epochs": result.num_epochs,
            "queue_size_mean": sum(size) / len(size),
            "queue_size_max": max(size),
            "queue_size_min": min(size),
            "latency_mean": sum(latency) / len(latency),
            "latency_max": max(latency),
            "latency_min": min(latency),
            "tasks_total": len(latency) + len(failed_latency),
            "tasks_failed": len(failed_latency),
            "fail_rate": len(failed_latency) / (len(latency) + len(failed_latency)),
        }
        return datum


def write_csv(path: str, data: list[dict]):
    with open(path, newline="", mode="w") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)


def experiment(num_clients: int):
    result = run_experiment(num_clients=num_clients, queue_size=12)
    datum = analyze_result(result)
    return num_clients, {"num_clients": num_clients, **datum}


def main():
    inputs = list(range(1, 40, 1))

    pool = multiprocessing.Pool()
    result = pool.imap_unordered(experiment, inputs)
    data = list(tqdm(result, total=len(inputs)))
    data.sort(key=itemgetter(0))

    write_csv("results/queue-size-12.csv", [datum for _, datum in data])


if __name__ == "__main__":
    raise SystemExit(main())
