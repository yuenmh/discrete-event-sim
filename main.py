import inspect
import types
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any, Coroutine, Generator, Protocol
from uuid import UUID, uuid4

from frozendict import frozendict


@dataclass(frozen=True, slots=True)
class RawMessage:
    args: tuple[Any, ...]
    kwargs: frozendict[str, Any]


@dataclass(frozen=True, slots=True)
class Addr:
    name: str


@dataclass(frozen=True, slots=True)
class Ref:
    uuid: UUID = field(default_factory=uuid4)


class Ctx:
    def __init__(self, state: dict[str, Any], self_addr: Addr, epoch_idx: int) -> None:
        self.state = state
        self.self = self_addr
        self.sent_messages: list[tuple[Addr, RawMessage]] = []
        self.epoch_idx = epoch_idx

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
    def __init__(self, branches: list[TransitionBranch] | None = None) -> None:
        self.branches = branches or []

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
        self, fuel: int, self_addr: Addr, epoch_idx: int
    ) -> list[tuple[Addr, RawMessage]]:
        ctx = Ctx(state=self.state, self_addr=self_addr, epoch_idx=epoch_idx)
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


class EventLoop:
    def __init__(self):
        self.nodes = dict[Addr, Node]()
        self.fuel_per_epoch = 10
        self.data_collection_paths: list[tuple[Addr, str]] = []
        self.data: list[DataPoint] = []
        self.epoch = 0

    def collect_data(self):
        dct = {}
        for addr, key in self.data_collection_paths:
            dct[(addr, key)] = self.nodes[addr].state.get(key)
        self.data.append(DataPoint(self.epoch, dct))

    def spawn(self, addr: Addr, transition: Transition, *args: Any, **kwargs: Any):
        self.nodes[addr] = Node(transition, dict())
        if args or kwargs:
            self.nodes[addr].inbox.append(RawMessage(args, frozendict(kwargs)))

    def run(self, epochs: int = 1):
        for _ in range(epochs):
            for addr, node in self.nodes.items():
                sent_messages = node.run(
                    fuel=self.fuel_per_epoch,
                    self_addr=addr,
                    epoch_idx=self.epoch,
                )
                for dest_addr, raw_msg in sent_messages:
                    self.nodes[dest_addr].inbox.append(raw_msg)
            self.collect_data()
            self.epoch += 1


def send(addr: Addr, *args: Any, **kwargs: Any):
    ctx().send(addr, *args, **kwargs)


def self() -> Addr:
    return ctx().self


async def wait(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
    return await ctx().wait(*args, **kwargs)


def log(**kwargs: Any):
    for k, v in kwargs.items():
        ctx().state[k] = v


async def ask(addr: Addr, method: Any, *args: Any, **kwargs: Any) -> Any:
    ref = Ref()
    send(addr, method, self(), ref, *args, **kwargs)
    result, *_ = await wait(ref)
    return result


def now() -> int:
    return ctx().epoch_idx


def main():
    class Add(object): ...

    class Loop(object): ...

    class Sleep(object): ...

    def make_timer():
        timer = Transition()

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

    def make_adder():
        adder = Transition()

        @adder.branch(Add)
        async def add(sender: Addr, ref: Ref, a: int, b: int):
            send(sender, ref, a + b)

        return adder

    def make_main(adder: Addr, timer: Addr):
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

            await ask(timer, Sleep, 1)

            state.i += 1

        return main

    event_loop = EventLoop()
    event_loop.spawn(Addr("adder"), make_adder())
    event_loop.spawn(Addr("timer"), make_timer(), Loop)
    event_loop.spawn(Addr("main"), make_main(Addr("adder"), Addr("timer")), Loop)
    event_loop.data_collection_paths.append((Addr("main"), "result"))
    event_loop.run(epochs=10)

    for data_point in event_loop.data:
        print(f"Epoch {data_point.epoch}: {data_point.data}")


if __name__ == "__main__":
    raise SystemExit(main())
