from dataclasses import dataclass
from typing import Any

from .sim import (
    Addr,
    Atom,
    Ref,
    SMBuilder,
    StateMachineBase,
    StateMachineInit,
    ask,
    handle,
    log,
    message,
    send,
)

__all__ = ["Ok", "Err", "Queue"]


class Ok(Atom): ...


class Err(Atom): ...


@dataclass
class Queue[T]:
    addr: Addr

    class Enqueue(Atom): ...

    class Dequeue(Atom): ...

    class Full(Atom): ...

    async def enqueue(self, item: T) -> bool:
        result = await ask(self.addr, Queue.Enqueue, item)
        return result is not Queue.Full

    async def dequeue(self) -> T:
        return await ask(self.addr, Queue.Dequeue)

    @classmethod
    def create(cls, max_size: int = 10) -> StateMachineInit:
        items = []
        waiting: list[tuple[Addr, Ref]] = []

        queue = SMBuilder()

        def log_size():
            log("queue size", size=len(items))

        @queue.handle(Queue.Enqueue)
        async def enqueue(sender: Addr, ref: Ref, item: Any):
            if len(items) >= max_size:
                send(sender, ref, Queue.Full)
            else:
                items.append(item)
                while waiting and items:
                    send(*waiting.pop(0), items.pop(0))
                log_size()
                send(sender, ref, Ok, hint="enqueued")

        @queue.handle(Queue.Dequeue)
        async def dequeue(sender: Addr, ref: Ref):
            if items:
                send(sender, ref, items.pop(0))
                log_size()
            else:
                waiting.append((sender, ref))

        return queue


class LaunchedStateMachine(StateMachineBase):
    class Start(Atom): ...

    __init_messages__ = (message(Start),)

    @handle(Start)
    async def _start(self):
        await self.start()

    async def start(self):
        """Override to implement the launched state machine's behavior."""
        raise NotImplementedError
