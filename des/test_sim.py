from dataclasses import dataclass

from .sim import (
    Addr,
    Atom,
    EventLoop,
    Ref,
    SMBuilder,
    ask,
    ask_timeout,
    launch,
    log,
    loop,
    send,
    sleep,
)
from .stdlib import Queue


def test_simple():
    class Add(Atom): ...

    def make_adder():
        adder = SMBuilder()

        @adder.handle(Add)
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
    event_loop.spawn(queue_addr, Queue.create(max_size=30))
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
