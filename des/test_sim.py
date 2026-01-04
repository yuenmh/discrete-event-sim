import warnings
from dataclasses import dataclass

from .sim import (
    Addr,
    Atom,
    EventLoop,
    Ref,
    SMBuilder,
    StateMachineBase,
    addr_of,
    ask,
    ask_timeout,
    handle,
    interface,
    launch,
    log,
    loop,
    rng,
    send,
    sleep,
    sleep_until,
    spawn,
    spawn_interface,
    stop,
)
from .stdlib import LaunchedStateMachine, Queue, Semaphore, WaitGroup


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
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        event_loop.run(epochs=50)

    assert any(
        log.msg == "timed out" for log in event_loop.logs if log.addr.name == "main"
    )


def test_class_based_counter():
    class Counter(StateMachineBase):
        def __init__(self, start: int = 0):
            self.counter = start

        @handle()
        async def inc(self):
            self.counter += 1

        @handle()
        async def _get(self, sender: Addr, ref: Ref):
            send(sender, ref, self.counter)

        async def get(self) -> int:
            return await ask(addr_of(self), Counter._get)

    counter_addr = Addr("counter")
    counter = interface(Counter, counter_addr)

    class Main(LaunchedStateMachine):
        async def start(self):
            print("started")

            for _ in range(20):
                counter.inc()
                await sleep(10)

            await sleep_until(100_000)

            value = await counter.get()
            assert value == 20
            log("done")

    event_loop = EventLoop()
    event_loop.spawn(counter_addr, Counter())
    event_loop.spawn(Addr("main"), Main())
    assert any(log.msg == "done" for log in event_loop.run(epochs=1_000_000).logs)


def test_stop_no_await():
    @launch
    async def main():
        i = 0
        while True:
            log(i=i)
            i += 1
            stop()

    event_loop = EventLoop()
    event_loop.spawn(Addr("main"), main)
    logs = event_loop.run(epochs=100).logs
    assert len(logs) == 1
    assert logs[0].data["i"] == 0


def test_stop_after_await():
    @launch
    async def main():
        i = 0
        while True:
            await sleep(1)
            log(i=i)
            i += 1
            stop()

    event_loop = EventLoop()
    event_loop.spawn(Addr("main"), main)
    logs = event_loop.run(epochs=100).logs
    assert len(logs) == 1
    assert logs[0].data["i"] == 0


def test_spawn_tasks():
    class BackgroundTask(LaunchedStateMachine):
        async def start(self):
            await sleep_until(20)
            log("foo")

    @launch
    async def main():
        for _ in range(10):
            spawn(BackgroundTask())

    event_loop = EventLoop()
    event_loop.spawn(Addr("main"), main)
    logs = event_loop.run(epochs=100).logs

    assert len(logs) == 10
    assert all(log.msg == "foo" and log.epoch == 20 for log in logs)


def test_spawn_and_join():
    class Worker(LaunchedStateMachine):
        def __init__(self, wg: WaitGroup):
            self.wg = wg

        async def start(self):
            await sleep(rng().randint(10, 20))
            self.wg.done()
            log("done")

    @launch
    async def main():
        wg = spawn_interface(WaitGroup())
        for _ in range(10):
            wg.add()
            spawn(Worker(wg))
        await wg.wait()
        log("all done")

    event_loop = EventLoop()
    event_loop.spawn(Addr("main"), main)
    logs = event_loop.run(epochs=100).logs

    assert any(log.msg == "all done" for log in logs)
    all_done_ts = next(log.epoch for log in logs if log.msg == "all done")
    max_done_ts = max(log.epoch for log in logs if log.msg == "done")
    assert all_done_ts >= max_done_ts


def test_semaphore():
    class Task(LaunchedStateMachine):
        def __init__(self, sem: Semaphore, wg: WaitGroup):
            self.sem = sem
            self.wg = wg

        async def start(self):
            await self.sem.acquire()
            log("acquired")
            await sleep(rng().randint(10, 20))
            self.sem.release()
            log("released")
            self.wg.done()

    @launch
    async def main():
        sem = spawn_interface(Semaphore(max_count=3))
        wg = spawn_interface(WaitGroup())
        for _ in range(100):
            wg.add()
            spawn(Task(sem, wg))
        await wg.wait()
        log("done")

    event_loop = EventLoop()
    event_loop.spawn(Addr("main"), main)
    logs = event_loop.run(epochs=10000).logs

    assert any(log.msg == "done" for log in logs)

    # at most 3 "acquired" logs should be active at any time
    current_holders = 0
    for log_entry in logs:
        if log_entry.msg == "acquired":
            current_holders += 1
            assert current_holders <= 3
        elif log_entry.msg == "released":
            current_holders -= 1
    assert current_holders == 0
