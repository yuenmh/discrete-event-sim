import csv
import multiprocessing
import os
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Literal,
    NamedTuple,
    Sequence,
)

from tqdm import tqdm

from des.sim import (
    Addr,
    Atom,
    EventLoop,
    Ref,
    RunResult,
    SMBuilder,
    StateMachineInit,
    Timeout,
    ask,
    ask_timeout,
    async_branch_handler,
    build_state_machine,
    launch,
    log,
    loop,
    now,
    rng,
    self,
    self_node,
    send,
    sleep,
)


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
        log("queue size", size=len(items))

    @queue.branch_handler(Enqueue)
    async def enqueue(sender: Addr, ref: Ref, item: Any):
        if len(items) >= max_size:
            send(sender, ref, QueueFull)
        else:
            items.append(item)
            while waiting and items:
                send(*waiting.pop(0), items.pop(0))
            log_size()
            send(sender, ref, Ok, hint="enqueued")

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


def constant_retry(sleep_time: int):
    def policy(_try_num: int) -> int:
        return sleep_time

    return policy


def run_experiment(
    num_clients: int = 10,
    num_clients_spike: int = 50,
    num_clients_after_spike: int = 30,
    num_workers: int = 4,
    queue_size: int = 14,
    num_epochs: int = 30000,
    num_retries: int = 3,
    retry_policy: Callable[[int], int] = constant_retry(20),
    spike_offset: int = 10000,
    spike_duration: int = 3000,
    submit_timeout: int = 30,
    work_time_range: tuple[int, int] = (5, 20),
    inter_task_sleep_range: tuple[int, int] = (10, 30),
):
    def make_worker(queue: Queue[tuple[Addr, Ref]]):
        @loop
        async def start():
            addr, ref = await queue.dequeue()
            log("start task", ref=ref)
            await sleep(rng().randrange(*work_time_range))
            log("finish task", ref=ref)
            send(addr, ref, Ok, hint="task_finished")

        return start

    queue_addrs = [Addr(f"queue-{i}") for i in range(num_workers)]

    load_balancer = SMBuilder()

    class SubmitWork(Atom): ...

    class Outstanding(NamedTuple):
        sender: Addr
        ref: Ref
        queue: Addr

    lb_outstanding: dict[Ref, Outstanding] = {}
    num_requests_sent = {addr: 0 for addr in queue_addrs}

    @load_balancer.branch_handler(SubmitWork)
    async def submit_work(sender: Addr, ref: Ref):
        worker_queue_addr = min(queue_addrs, key=lambda addr: num_requests_sent[addr])
        send(worker_queue_addr, Enqueue, self(), ref, (sender, ref))
        num_requests_sent[worker_queue_addr] += 1
        lb_outstanding[ref] = Outstanding(
            sender=sender, ref=ref, queue=worker_queue_addr
        )

    @load_balancer.branch_handler(..., QueueFull)
    async def handle_queue_full(ref: Ref):
        outstanding = lb_outstanding.pop(ref, None)
        assert outstanding is not None, "Received QueueFull for unknown submission"
        log("queue full", queue=outstanding.queue)
        send(outstanding.sender, outstanding.ref, Err, hint="work_failed")

    @load_balancer.branch_handler(..., Ok)
    async def handle_ok(ref: Ref):
        outstanding = lb_outstanding.pop(ref, None)
        assert outstanding is not None, "Received Ok for unknown submission"

    lb_addr = Addr("load_balancer")

    async def submit_work_wrapper(
        n_retries: int = 3,
        timeout: int | None = None,
        policy: Callable[[int], int] = constant_retry(10),
        deadline: int | None = None,
    ) -> bool:
        submission_id = Ref()
        for try_num in range(n_retries):
            if deadline is not None and now() >= deadline:
                log(
                    "deadline exceeded",
                    deadline=deadline,
                    try_num=try_num,
                    submission_id=submission_id,
                )
                return False
            if timeout:
                try:
                    result = await ask_timeout(
                        timeout,
                        lb_addr,
                        SubmitWork,
                    )
                except Timeout:
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

    async def perform_work(deadline: int | None = None):
        start = now()
        success = await submit_work_wrapper(
            n_retries=num_retries,
            timeout=submit_timeout,
            policy=retry_policy,
            deadline=deadline,
        )
        if success:
            log("finished", latency=now() - start, start=start)
        else:
            log("failed", latency=now() - start)

    clients: list[StateMachineInit] = []

    def make_normal_client(client_ix: int):
        @launch
        async def client():
            log("client started", client_ix=client_ix)
            while True:
                await perform_work()
                await sleep(rng().randrange(*inter_task_sleep_range))

        return client

    def make_spike_client(client_ix: int):
        @launch
        async def client():
            await sleep(spike_offset + rng().randrange(0, 300))
            log("client started", client_ix=client_ix)
            while True:
                await perform_work(
                    deadline=spike_offset + spike_duration + rng().randrange(0, 300)
                )
                if now() >= spike_offset + spike_duration:
                    break
                await sleep(rng().randrange(*inter_task_sleep_range))
            log("client exited", client_ix=client_ix)
            self_node().stopped = True

        return client

    def make_added_client(client_ix: int):
        @launch
        async def client():
            await sleep(spike_offset + rng().randrange(0, 300))
            log("client started", client_ix=client_ix)
            while True:
                await perform_work()
                await sleep(rng().randrange(*inter_task_sleep_range))

        return client

    for client_ix in range(
        max(num_clients, num_clients_spike, num_clients_after_spike)
    ):
        if client_ix < num_clients:
            clients.append(make_normal_client(client_ix))
        elif client_ix < num_clients_after_spike:
            clients.append(make_added_client(client_ix))
        elif client_ix < num_clients_spike:
            clients.append(make_spike_client(client_ix))

    event_loop = EventLoop()
    for i, queue_addr in enumerate(queue_addrs):
        event_loop.spawn(queue_addr, make_queue(max_size=queue_size))
        event_loop.spawn(Addr(f"worker-{i}"), make_worker(Queue(queue_addr)))
    event_loop.spawn(lb_addr, load_balancer)
    for i, client in enumerate(clients):
        event_loop.spawn(Addr(f"client-{i}"), client)

    result = event_loop.run(epochs=num_epochs)
    return result


def analyze_result(result: RunResult):
    import numpy as np
    import polars as pl

    with result.logs_sqlite() as conn:
        client_starts = [
            t
            for (t,) in conn.execute(
                "select epoch from log where msg='client started'"
            ).fetchall()
        ]
        client_stops = [
            t
            for (t,) in conn.execute(
                "select epoch from log where msg='client exited'"
            ).fetchall()
        ]
        start_counts = np.bincount(client_starts, minlength=result.num_epochs)
        stop_counts = np.bincount(client_stops, minlength=result.num_epochs)
        num_clients = np.cumsum(start_counts - stop_counts)

        task_latency = [
            (t, lt)
            for t, lt in conn.execute(
                "select data->>'start', data->>'latency' from log where msg='finished'"
            ).fetchall()
        ]
        latency_df = pl.DataFrame(task_latency, schema=["time", "value"], orient="row")
        averages = latency_df.group_by("time").agg(pl.col("value").mean())
        latency_result_df = (
            pl.DataFrame({"time": range(result.num_epochs)})
            .join(averages, on="time", how="left")
            .with_columns(pl.col("value").forward_fill().fill_null(0))
        )

        smoothing_window = 300
        smoothed_latency = np.convolve(
            latency_result_df["value"].to_numpy(),
            np.ones(smoothing_window) / smoothing_window,
            mode="same",
        )

        queue_size = [
            (t, sz)
            for t, sz in conn.execute(
                "select epoch, data->>'size' from log where msg='queue size'"
            ).fetchall()
        ]
        queue_size_df = pl.DataFrame(queue_size, schema=["time", "value"], orient="row")
        averages = queue_size_df.group_by("time").agg(pl.col("value").mean())
        queue_size_result_df = (
            pl.DataFrame({"time": range(result.num_epochs)})
            .join(averages, on="time", how="left")
            .with_columns(pl.col("value").forward_fill().fill_null(0))
        )
        smoothed_queue_size = np.convolve(
            queue_size_result_df["value"].to_numpy(),
            np.ones(smoothing_window) / smoothing_window,
            mode="same",
        )

        successes = [
            t
            for (t,) in conn.execute(
                "select epoch from log where msg='finished'"
            ).fetchall()
        ]
        failures = [
            t
            for (t,) in conn.execute(
                "select epoch from log where msg='failed'"
            ).fetchall()
        ]
        success_counts = np.bincount(successes, minlength=result.num_epochs)
        failure_counts = np.bincount(failures, minlength=result.num_epochs)
        total_successes = np.cumsum(success_counts)
        total_failures = np.cumsum(failure_counts)

        df = pl.DataFrame(
            {
                "time": range(result.num_epochs),
                "num_clients": num_clients,
                "task_latency": latency_result_df["value"],
                "smoothed_task_latency": smoothed_latency,
                "queue_size": queue_size_result_df["value"],
                "smoothed_queue_size": smoothed_queue_size,
                "total_successes": total_successes,
                "total_failures": total_failures,
            }
        )
        return df


def write_csv(path: str, data: list[dict]):
    with open(path, newline="", mode="w") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)


class Runner:
    def __init__(self):
        self.runs = []

    def trial(self, values: Sequence[Any], name: str | None = None):
        def decorator(fn: Callable[..., Any]):
            trial_name = name or fn.__name__.replace("_", "-")
            self.runs.append((trial_name, fn, values))
            return fn

        return decorator

    def _runner_fn(self, args: tuple[str, Callable[..., Any], Any, str]):
        import polars as pl

        _name, fn, input, output_path = args
        result = fn(input)
        if isinstance(result, pl.DataFrame):
            result.write_csv(output_path, include_header=True)
        elif isinstance(result, list):
            write_csv(output_path, result)
        else:
            raise TypeError("Unsupported result type")

    def run_all(self, force: bool = False, concurrent: bool = True):
        runs = []
        for name, fn, inputs in self.runs:
            for input in inputs:
                output_path = f"results/{name}-{input}.csv"
                if os.path.exists(output_path) and not force:
                    print(f"Skip {name} with input {input} (already exists)")
                    continue
                runs.append((name, fn, input, output_path))

        print("To run")
        for name, *_ in runs:
            print(f"  {name}")

        if concurrent:
            pool = multiprocessing.Pool()
            list(tqdm(pool.imap_unordered(self._runner_fn, runs), total=len(runs)))
        else:
            for run in tqdm(runs):
                self._runner_fn(run)


runner = Runner()


def run_single_server_case(queue_size: int):
    return run_experiment(
        num_workers=1,
        queue_size=queue_size,
        num_retries=100_000,
        num_epochs=40_000,
        spike_offset=8000,
        spike_duration=2000,
        num_clients=3,
        num_clients_spike=5,
        num_clients_after_spike=3,
        submit_timeout=83,
        work_time_range=(25, 27),
        inter_task_sleep_range=(49, 52),
        retry_policy=constant_retry(2),
    )


@runner.trial([1, 10])
def single_server_vary_queue_size(qs):
    return analyze_result(run_single_server_case(queue_size=qs))


@runner.trial(["control", "test"])
def multiple_servers(version: Literal["control", "test"]):
    return analyze_result(
        run_experiment(
            num_workers=4,
            queue_size=10,
            num_retries=100_000,
            num_epochs=60_000,
            spike_offset=12000 if version == "test" else 100_000,
            spike_duration=2000,
            num_clients=12,
            num_clients_spike=20,
            num_clients_after_spike=12,
            submit_timeout=83,
            work_time_range=(25, 27),
            inter_task_sleep_range=(49, 52),
            retry_policy=constant_retry(2),
        )
    )


def main():
    runner.run_all(force=False, concurrent=True)


if __name__ == "__main__":
    raise SystemExit(main())
