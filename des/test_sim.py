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
    now,
    rng,
    send,
    sleep,
    sleep_until,
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


def test_clone_same_state():
    def create_counter():
        counter = SMBuilder()

        value = 0

        @counter.handle("inc")
        async def handle_inc():
            nonlocal value
            value += 1

        @counter.handle("get")
        async def handle_get(sender: Addr, ref: Ref):
            send(sender, ref, value)

        return counter

    counter_addr = Addr("counter")

    def create_main():
        @launch
        async def main():
            for _ in range(20):
                send(counter_addr, "inc")
                await sleep(10)

            await sleep_until(100_000)

            value = await ask(counter_addr, "get")
            assert value == 20
            log("done")

        return main

    event_loop = EventLoop()
    event_loop.spawn(counter_addr, create_counter)
    event_loop.spawn(Addr("main"), create_main)
    event_loop.run(epochs=10_000)

    el1 = event_loop.clone()
    el2 = event_loop.clone()

    res1 = el1.run(epochs=200_000)
    assert res1.logs[-1].msg == "done"

    res2 = el2.run(epochs=200_000)
    assert res2.logs[-1].msg == "done"

    assert res1.logs == res2.logs


def test_clone_divergent_state():
    def create_thing():
        @launch
        async def f():
            count = 0
            last = now()
            while True:
                await sleep(rng().randint(5, 15))
                count += now() - last
                last = now()
                log(count=count)

        return f

    event_loop = EventLoop()
    event_loop.spawn(Addr("thing"), create_thing)
    event_loop.run(epochs=2000)

    el1 = event_loop.clone()
    el2 = event_loop.clone()
    el1.find("thing").seed_rng(42)
    el2.find("thing").seed_rng(42)

    res_init = event_loop.run(epochs=4000)
    res1 = el1.run(epochs=4000)
    res2 = el2.run(epochs=4000)

    assert res_init.logs != res1.logs
    assert res1.logs == res2.logs
