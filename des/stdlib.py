from collections import deque
from dataclasses import dataclass
from typing import Any

from .sim import (
    Addr,
    Atom,
    Ref,
    SMBuilder,
    StateMachineBase,
    StateMachineInit,
    addr_of,
    ask,
    handle,
    log,
    message,
    send,
    stop,
    this,
    wait,
    wait_timeout,
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
        stop()

    async def start(self):
        """Override to implement the launched state machine's behavior."""
        raise NotImplementedError


class WaitGroup(StateMachineBase):
    def __init__(self, count: int = 0):
        self._count = count
        self._waiters: list[tuple[Addr, Ref]] = []

    @handle()
    async def _change(self, delta: int):
        new_count = self._count + delta
        if new_count <= 0:
            self._count = 0
            for addr, ref in self._waiters:
                send(addr, ref, Ok)
            self._waiters.clear()
        else:
            self._count = new_count

    def add(self, delta: int = 1):
        send(addr_of(self), WaitGroup._change, delta)

    def done(self):
        self.add(-1)

    @handle()
    async def _wait(self, sender: Addr, ref: Ref):
        if self._count == 0:
            send(sender, ref, Ok)
        else:
            self._waiters.append((sender, ref))

    async def wait(self):
        await ask(addr_of(self), WaitGroup._wait)


# TODO: fix issue with context manager not having access to the event loop contextvar
class Semaphore(StateMachineBase):
    def __init__(self, max_count: int):
        self.max_count = max_count
        self.current_count = 0
        self.waiting: deque[tuple[Addr, Ref]] = deque()

    @handle()
    async def _acquire(self, requester: Addr, ref: Ref):
        if self.current_count < self.max_count:
            self.current_count += 1
            send(requester, ref, hint="semaphore granted instantly")
        else:
            self.waiting.append((requester, ref))

    async def acquire(self, timeout: int | None = None):
        self._acquire(this(), ref := Ref())
        if timeout is not None:
            await wait_timeout(timeout, ref)
        else:
            await wait(ref)

    @handle()
    async def release(self):
        if self.waiting:
            requester, ref = self.waiting.popleft()
            send(requester, ref, hint="semaphore granted after wait")
        else:
            self.current_count -= 1
