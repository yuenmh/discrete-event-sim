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
    Protocol,
    Sequence,
)
from uuid import UUID, uuid4

from frozendict import frozendict


@dataclass(frozen=True, slots=True)
class RawMessage:
    args: tuple[Any, ...]
    kwargs: frozendict[str, Any]


def raw_message(*args: Any, **kwargs: Any) -> RawMessage:
    return RawMessage(args, frozendict(kwargs))


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
        self.sent_messages: list[tuple[Addr, RawMessage]] = []
        self.epoch_idx = epoch_idx
        self._event_loop = event_loop

    def send(self, addr: Addr, *args: Any, **kwargs: Any) -> None:
        self.sent_messages.append((addr, RawMessage(args, frozendict(kwargs))))

    @types.coroutine
    def wait(
        self, *args: Any, **kwargs: Any
    ) -> Generator[CoBranchYield, RawMessage, tuple[Any, ...]]:
        msg = yield CoBranchYield(args, kwargs)
        return *msg.args, msg.kwargs


class TransitionFn(Protocol):
    def __call__(self, ctx: Ctx, msg: RawMessage) -> "Transition": ...


class TransitionImplFn(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> "Transition": ...


def _wrap_normal_transition_fn(fn: TransitionImplFn) -> TransitionFn:
    sig = inspect.signature(fn)

    def wrapped(ctx: Ctx, msg: RawMessage) -> Transition:
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
    ) -> Coroutine[CoBranchYield, RawMessage | None, None]: ...


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

    def wrapped(ctx: Ctx, msg: RawMessage) -> Transition:
        bound = sig.bind(*msg.args, **msg.kwargs)
        with scoped_ctx(ctx):
            co = fn(*bound.args, **bound.kwargs)
            try:
                yield_params = co.send(None)
            except StopIteration:
                return parent_transition
        t = Transition()

        @t.add_branch(*yield_params.match_args, **yield_params.match_kwargs)
        def resume(ctx: Ctx, msg: RawMessage):
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
        init_messages: Sequence[RawMessage] = (),
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

    def find_matching_branch(self, msg: RawMessage) -> TransitionBranch | None:
        for branch in self.branches:
            if branch.matches(msg):
                return branch
        return None

    def find_message(
        self, msgs: list[RawMessage]
    ) -> tuple[int, RawMessage, TransitionBranch] | None:
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

    def matches(self, msg: RawMessage) -> bool:
        if (
            len(msg.args) >= len(self.match_args)
            and msg.args[: len(self.match_args)] == self.match_args
        ):
            for key, value in self.match_kwargs.items():
                if key not in msg.kwargs or msg.kwargs[key] != value:
                    return False
            return True
        return False

    def call(self, ctx: Ctx, msg: RawMessage) -> Transition:
        return self.fn(ctx, msg)


class Node:
    def __init__(self, transition: Transition, state: Any):
        self.root_transition = transition
        self.current_transition = transition
        self.state = state
        self.inbox: list[RawMessage] = []

    def run(
        self, fuel: int, self_addr: Addr, epoch_idx: int, event_loop: EventLoop
    ) -> list[tuple[Addr, RawMessage]]:
        ctx = Ctx(
            state=self.state,
            self_addr=self_addr,
            epoch_idx=epoch_idx,
            event_loop=event_loop,
        )
        for _ in range(fuel):
            if (found := self.current_transition.find_message(self.inbox)) is None:
                break
            i, msg, branch = found
            self.inbox.pop(i)
            self.current_transition = branch.call(ctx, msg)
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

    def collect_data(self):
        dct = {}
        for addr, key in self.data_collection_paths:
            dct[(addr, key)] = self.nodes[addr].state.get(key)
        self.data.append(DataPoint(self.epoch, dct))

    def spawn(
        self,
        addr: Addr,
        transition: Transition,
        *init_messages: RawMessage,
    ):
        self.nodes[addr] = Node(transition, dict())
        self.nodes[addr].inbox.extend(transition.init_messages)
        if init_messages is not None:
            self.nodes[addr].inbox.extend(init_messages)
        self.rngs[addr] = Random(hashlib.sha256(addr.name.encode()).digest())

    def spawn_spec(self, *specs: NodeSpec):
        for spec in specs:
            self.spawn(spec.addr, spec.transition())

    def run(self, epochs: int = 1):
        for _ in range(epochs):
            sent_messages: list[tuple[Addr, RawMessage]] = []
            for addr, node in self.nodes.items():
                sent_messages.extend(
                    node.run(
                        fuel=self.fuel_per_epoch,
                        self_addr=addr,
                        epoch_idx=self.epoch,
                        event_loop=self,
                    )
                )
            for dest_addr, raw_msg in sent_messages:
                try:
                    self.nodes[dest_addr].inbox.append(raw_msg)
                except KeyError:
                    raise RuntimeError(f"Unknown address: {dest_addr}")
            self.collect_data()
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
    def transition(self) -> Transition: ...


class TimerSpec:
    addr = _TIMER_ADDR

    def transition(_self) -> Transition:
        timer = Transition(init_messages=[raw_message(Loop)])

        waiting_tasks: dict[Ref, tuple[Addr, int]] = {}

        @timer.branch(Sleep)
        async def sleep(sender: Addr, ref: Ref, duration: int):
            wake_time = now() + duration
            waiting_tasks[ref] = (sender, wake_time)

        @timer.branch(Loop)
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

        return timer


RUNTIME_SPECS = (TimerSpec(),)


class TransitionCoroutineFnReturn(Protocol):
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> Coroutine[CoBranchYield, RawMessage | None, Any]: ...


def responder(fn: TransitionCoroutineFnReturn):
    async def wrapped(sender: Addr, ref: Ref, *args: Any, **kwargs: Any):
        result = await fn(*args, **kwargs)
        send(sender, ref, result)

    return wrapped


def loop(fn: Callable[..., Awaitable[None]]) -> Transition:
    t = Transition(init_messages=[raw_message(Loop)])

    @t.branch(Loop)
    async def start():
        send(self(), Loop)
        await fn()

    return t


def launch(fn: Callable[..., Awaitable[None]]) -> Transition:
    t = Transition(init_messages=[raw_message(None)])

    @t.branch(None)
    async def start():
        await fn()

    return t


def test_simple():
    class Add(Atom): ...

    def make_adder():
        adder = Transition()

        @adder.branch(Add)
        @responder
        async def add(a: int, b: int):
            return a + b

        return adder

    def make_main(adder: Addr):
        @dataclass
        class State:
            i: int = 0

        state = State()

        main = Transition()

        @main.branch(Loop)
        async def start():
            send(self(), Loop)

            result = await ask(adder, Add, state.i, 10)
            log(result=result)

            await sleep(1)

            state.i += 1

        return main

    event_loop = EventLoop()
    event_loop.spawn_spec(*RUNTIME_SPECS)
    event_loop.spawn(Addr("adder"), make_adder())
    event_loop.spawn(Addr("main"), make_main(Addr("adder")), raw_message(Loop))
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
