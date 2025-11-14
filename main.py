import inspect
from dataclasses import dataclass, field, replace
from typing import Any, Protocol
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
    def __init__(self, state: dict[str, Any], self_addr: Addr) -> None:
        self.state = state
        self.self = self_addr
        self.sent_messages: list[tuple[Addr, RawMessage]] = []

    def send(self, addr: Addr, *args: Any, **kwargs: Any) -> None:
        self.sent_messages.append((addr, RawMessage(args, frozendict(kwargs))))


class TransitionFn(Protocol):
    def __call__(self, ctx: Ctx, *args: Any, **kwargs: Any) -> "Transition": ...


class Transition:
    def __init__(self, branches: list[TransitionBranch] | None = None) -> None:
        self.branches = branches or []

    def branch(self, *args: Any, **kwargs: Any):
        def add_branch(fn: TransitionFn):
            self.branches.append(TransitionBranch(args, kwargs, fn))

        return add_branch

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


class TransitionBranch[State]:
    def __init__(
        self,
        match_args: tuple[Any],
        match_kwargs: dict[str, Any],
        fn: TransitionFn,
    ) -> None:
        self.match_args = match_args
        self.match_kwargs = match_kwargs
        self.fn = fn
        self.sig = inspect.signature(fn)

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
        bound = self.sig.bind(ctx, *msg.args, **msg.kwargs)
        return self.fn(*bound.args, **bound.kwargs)


def wait[State](*args: Any, **kwargs: Any):
    def decorator(fn: TransitionFn):
        t = Transition()
        t.branch(*args, **kwargs)(fn)
        return t

    return decorator


class Node:
    def __init__(self, transition: Transition, state: Any):
        self.root_transition = transition
        self.current_transition = transition
        self.state = state
        self.inbox: list[RawMessage] = []

    def run(self, fuel: int, self_addr: Addr) -> list[tuple[Addr, RawMessage]]:
        ctx = Ctx(state=self.state, self_addr=self_addr)
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
                sent_messages = node.run(self.fuel_per_epoch, addr)
                for dest_addr, raw_msg in sent_messages:
                    self.nodes[dest_addr].inbox.append(raw_msg)
            self.collect_data()
            self.epoch += 1


def main():
    class Add(object): ...

    class Loop(object): ...

    def make_adder():
        adder = Transition()

        @adder.branch(Add)
        def add(ctx: Ctx, sender: Addr, ref: Ref, a: int, b: int):
            ctx.send(sender, ref, a + b)

            return adder

        return adder

    def make_main(adder: Addr):
        main = Transition()

        @dataclass
        class State:
            i: int = 0

        state = State()

        @main.branch(Loop)
        def start(ctx: Ctx):
            ctx.send(ctx.self, Loop)

            ref = Ref()
            ctx.send(adder, Add, ctx.self, ref, 2, state.i)

            @wait(ref)
            def recv_result(ctx: Ctx, result: int):
                ctx.state["result"] = result

                state.i += 1

                return main

            return recv_result

        return main

    event_loop = EventLoop()
    event_loop.spawn(Addr("adder"), make_adder())
    event_loop.spawn(Addr("main"), make_main(Addr("adder")), Loop)
    event_loop.data_collection_paths.append((Addr("main"), "result"))
    event_loop.run(epochs=10)

    for data_point in event_loop.data:
        print(f"Epoch {data_point.epoch}: {data_point.data}")


if __name__ == "__main__":
    raise SystemExit(main())
