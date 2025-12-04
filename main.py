import argparse
import csv
import multiprocessing
import re
from typing import (
    Any,
    Callable,
    Literal,
    NamedTuple,
    Sequence,
)

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
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
    launch,
    log,
    loop,
    now,
    rng,
    self,
    send,
    sleep,
    stop,
    wait_timeout,
)
from des.stdlib import Err, Ok, Queue


def constant_retry(wait_time: int):
    def policy(_try_num: int) -> int:
        return wait_time

    return policy


def linear_backoff_retry(base_wait_time: int, per_try: int):
    def policy(try_num: int) -> int:
        return base_wait_time + (per_try * try_num)

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
    work_time: tuple[int, int] = (5, 2),
    inter_task_sleep: tuple[int, int] = (10, 3),
    added_client_wake_range: int = 300,
):
    def make_worker(queue: Queue[tuple[Addr, Ref]]):
        @loop
        async def start():
            addr, ref = await queue.dequeue()
            log("start task", ref=ref)
            await sleep(work_time[0] + int(rng().expovariate(1.0 / work_time[1])))
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

    @load_balancer.handle(SubmitWork)
    async def submit_work(sender: Addr, ref: Ref):
        worker_queue_addr = min(queue_addrs, key=lambda addr: num_requests_sent[addr])
        send(worker_queue_addr, Queue.Enqueue, self(), ref, (sender, ref))
        num_requests_sent[worker_queue_addr] += 1
        lb_outstanding[ref] = Outstanding(
            sender=sender, ref=ref, queue=worker_queue_addr
        )

    @load_balancer.handle(..., Queue.Full)
    async def handle_queue_full(ref: Ref):
        outstanding = lb_outstanding.pop(ref, None)
        assert outstanding is not None, "Received QueueFull for unknown submission"
        log("queue full", queue=outstanding.queue)
        send(outstanding.sender, outstanding.ref, Err, hint="work_failed")

    @load_balancer.handle(..., Ok)
    async def handle_ok(ref: Ref):
        outstanding = lb_outstanding.pop(ref, None)
        assert outstanding is not None, "Received Ok for unknown submission"

    lb_addr = Addr("load_balancer")

    async def submit_work_wrapper(
        timeout: int,
        n_retries: int = 3,
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
            timeout=submit_timeout,
            n_retries=num_retries,
            policy=retry_policy,
            deadline=deadline,
        )
        if success:
            log("finished", latency=now() - start, start=start)
        else:
            log("failed", latency=now() - start)

    clients: list[StateMachineInit] = []

    async def sleep_between_tasks():
        await sleep(
            inter_task_sleep[0] + int(rng().expovariate(1.0 / inter_task_sleep[1]))
        )

    def make_normal_client(client_ix: int):
        @launch
        async def client():
            log("client started", client_ix=client_ix)
            while True:
                await perform_work()
                await sleep_between_tasks()

        return client

    def make_spike_client(client_ix: int):
        @launch
        async def client():
            wake_offset = rng().randrange(0, added_client_wake_range)
            await sleep(spike_offset + wake_offset)
            log("client started", client_ix=client_ix)
            while True:
                await perform_work(deadline=spike_offset + spike_duration + wake_offset)
                if now() >= spike_offset + spike_duration:
                    break
                await sleep_between_tasks()
            log("client exited", client_ix=client_ix)
            stop()

        return client

    def make_added_client(client_ix: int):
        @launch
        async def client():
            await sleep(spike_offset + rng().randrange(0, added_client_wake_range))
            log("client started", client_ix=client_ix)
            while True:
                await perform_work()
                await sleep_between_tasks()

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
        event_loop.spawn(queue_addr, Queue.create(max_size=queue_size))
        event_loop.spawn(Addr(f"worker-{i}"), make_worker(Queue(queue_addr)))
    event_loop.spawn(lb_addr, load_balancer)
    for i, client in enumerate(clients):
        event_loop.spawn(Addr(f"client-{i}"), client)

    result = event_loop.run(epochs=num_epochs)
    return result


def analyze_result(result: RunResult, smoothing_window: int = 300):
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

    def run_all(
        self,
        concurrent: bool = True,
        filter: re.Pattern[str] | None = None,
    ):
        runs = []
        for name, fn, inputs in self.runs:
            for input in inputs:
                output_path = f"results/{name}-{input}.csv"
                runs.append((name, fn, input, output_path))

        if filter is not None:
            runs = [run for run in runs if filter.match(run[0])]

        print("To run")
        for name, _, input, *_ in runs:
            print(f"  {name} - {input}")

        if concurrent:
            pool = multiprocessing.Pool()
            list(tqdm(pool.imap_unordered(self._runner_fn, runs), total=len(runs)))
        else:
            for run in tqdm(runs):
                self._runner_fn(run)


runner = Runner()


@runner.trial([1, 10])
def single_server_vary_queue_size(qs):
    return analyze_result(
        run_experiment(
            num_workers=1,
            queue_size=qs,
            num_retries=100_000,
            num_epochs=60_000,
            spike_offset=8000,
            spike_duration=2000,
            num_clients=3,
            num_clients_spike=5,
            num_clients_after_spike=3,
            work_time=(22, 5),
            inter_task_sleep=(50, 5),
            submit_timeout=83,
            retry_policy=constant_retry(wait_time=2),
        )
    )


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
            work_time=(22, 5),
            inter_task_sleep=(50, 5),
            submit_timeout=83,
            retry_policy=constant_retry(wait_time=2),
        )
    )


@runner.trial(["control", "test-linear", "test-const"])
def linear_backoff(version: Literal["control", "test-linear", "test-const"]):
    return analyze_result(
        run_experiment(
            num_workers=1,
            queue_size=20,
            num_retries=100_000,
            num_epochs=60_000,
            spike_offset=12000 if "test" in version else 100_000,
            spike_duration=2000,
            num_clients=12,
            num_clients_spike=20,
            num_clients_after_spike=12,
            work_time=(100, 7),
            inter_task_sleep=(2, 2),
            submit_timeout=12 * 110 + 70,
            retry_policy=constant_retry(0)
            if version == "test-const"
            else linear_backoff_retry(base_wait_time=0, per_try=1),
        )
    )


@runner.trial(("control", "test"))
def large_timescale(version: str):
    return analyze_result(
        run_experiment(
            num_workers=1,
            queue_size=20,
            num_retries=100_000,
            num_epochs=500_000,
            spike_offset=100_000,
            spike_duration=30_000,
            num_clients=5,
            num_clients_spike=8 if version == "test" else 5,
            num_clients_after_spike=5,
            work_time=(1_000, 500),
            inter_task_sleep=(0, 200),
            submit_timeout=5 * 2_000,
            retry_policy=constant_retry(200),
        ),
        smoothing_window=1000,
    )


def smooth_series(series: pl.Series, avg_window: int) -> pl.Series:
    return pl.Series(
        name=series.name,
        values=np.convolve(
            series.to_numpy(),
            np.ones(avg_window) / avg_window,
        )[: len(series)],
    )


def experiment2():
    work_time = 1000
    work_time_rand = 100
    timeout = 1800
    retry_delay = 100
    inter_sleep = 4000
    full_delay = 12000

    trigger_delay = 1_000_000

    this = self

    queue_addr = Addr("queue")
    queue = Queue[tuple[Addr, Ref]](queue_addr)

    class Worker:
        def __init__(self, addr: Addr):
            self.queue_addr = addr

        async def submit(self, ref: Ref, timeout: int) -> bool:
            log("submit", ref=ref)
            if await ask(self.queue_addr, Queue.Enqueue, (this(), ref)) is Queue.Full:
                log("fail", ref=ref, reason="full")
                await sleep(full_delay)
                return False
            try:
                await wait_timeout(timeout, ref)
            except Timeout:
                log("fail", ref=ref, reason="timeout")
                return False
            return True

    worker = Worker(queue_addr)

    @launch
    async def worker_process():
        while True:
            addr, ref = await queue.dequeue()
            await sleep(
                work_time - work_time_rand + int(rng().expovariate(1 / work_time_rand))
            )
            send(addr, ref, hint="done")

    @launch
    async def client():
        while True:
            retries = 0
            start = now()
            ref = Ref()
            while not await worker.submit(ref, timeout=timeout):
                retries += 1
                log("retry", ref=ref, retries=retries)
                await sleep(retry_delay)
            log("done", ref=ref, retries=retries, start=start, duration=now() - start)
            await sleep(inter_sleep)

    @launch
    async def client2():
        await sleep(200)
        while True:
            retries = 0
            start = now()
            ref = Ref()
            while not await worker.submit(ref, timeout=timeout):
                retries += 1
                log("retry", ref=ref, retries=retries)
                await sleep(retry_delay)
            log("done", ref=ref, retries=retries, start=start, duration=now() - start)
            await sleep(inter_sleep)

    @launch
    async def trigger():
        # return
        await sleep(trigger_delay)
        for _ in range(20):
            await worker.submit(Ref(), timeout=50)

    el = EventLoop()
    el.spawn(queue_addr, Queue.create(max_size=20))
    el.spawn(Addr("worker"), worker_process)
    el.spawn(Addr("client"), client)
    el.spawn(Addr("client2"), client2)
    el.spawn(Addr("trigger"), trigger)
    result = el.run(epochs=2_000_000)

    avg_window = 10
    task_times = (
        pl.DataFrame(
            [
                (
                    entry.data["start"] // 1000,
                    entry.data["duration"],
                    entry.data["retries"],
                )
                for entry in result.logs
                if entry.msg == "done" and entry.addr.name.startswith("client")
            ],
            schema=["epoch", "duration", "retries"],
            orient="row",
        )
        .group_by("epoch", maintain_order=True)
        .agg(pl.col("*").mean())
    )
    task_times = task_times.with_columns(
        pl.Series(
            name="duration",
            values=np.convolve(
                task_times["duration"].to_numpy(),
                np.ones(avg_window) / avg_window,
            )[: len(task_times)],
        )
    )
    task_times = task_times.with_columns(
        pl.arange(0, len(task_times)).alias("completed")
    )

    goodput = (
        pl.DataFrame()
        .with_columns(
            pl.arange(0, max(e.epoch for e in result.logs) // 1000 + 1).alias("epoch")
        )
        .join(task_times, on="epoch", how="left", maintain_order="left")
        .select("epoch", "completed")
        .with_columns(pl.col("completed").fill_null(strategy="forward").fill_null(0))
    )
    goodput = goodput.with_columns(
        pl.Series(
            name="completed",
            values=np.convolve(
                goodput["completed"].to_numpy(),
                np.ones(avg_window * 3) / (avg_window * 3),
            )[: len(goodput)],
        )
    )
    goodput = goodput.with_columns(
        pl.col("completed").diff().fill_null(0).alias("goodput")
    )

    queue_size = (
        pl.DataFrame(
            [
                (entry.epoch // 1000, entry.data["size"])
                for entry in result.logs
                if entry.msg == "queue size"
            ],
            schema=["epoch", "size"],
            orient="row",
        )
        .group_by("epoch", maintain_order=True)
        .agg(pl.col("size").mean())
    )
    queue_size = queue_size.with_columns(
        pl.Series(
            name="size",
            values=np.convolve(
                queue_size["size"].to_numpy(),
                np.ones(avg_window) / avg_window,
            )[: len(queue_size)],
        )
    )

    sends = (
        pl.DataFrame(
            [
                (entry.epoch // 10000)
                for entry in result.logs
                if entry.addr.name.startswith("client") and entry.msg == "submit"
            ],
            schema=["epoch"],
            orient="row",
        )
        .group_by("epoch", maintain_order=True)
        .agg(pl.len().alias("sends"))
        .with_columns(pl.col("epoch") * 10)
    )

    full = (
        pl.DataFrame(
            [
                (entry.epoch // 1000)
                for entry in result.logs
                if entry.data.get("reason") == "full"
            ],
            schema=["epoch"],
            orient="row",
        )
        .group_by("epoch", maintain_order=True)
        .agg(pl.len().alias("fulls"))
    )

    retries = (
        pl.DataFrame(
            [(entry.epoch // 1000) for entry in result.logs if entry.msg == "retry"],
            schema=["epoch"],
            orient="row",
        )
        .group_by("epoch", maintain_order=True)
        .agg(pl.len().alias("retries"))
    )
    retries = retries.with_columns(
        smooth_series(retries.get_column("retries"), avg_window=avg_window)
    )

    fig = plt.figure(figsize=(18, 8))
    fig.suptitle("Metrics")
    fig.tight_layout()

    grid = (2, 3)

    # ax1 = fig.add_subplot(*grid, 1)
    # ax1.set_title("Average Task Duration")
    # ax1.plot(task_times["epoch"], task_times["duration"], linewidth=1)

    ax2 = fig.add_subplot(*grid, 2)
    ax2.set_title("Total Completed Tasks")
    ax2.plot(goodput["epoch"], goodput["completed"], linewidth=1)

    ax3 = fig.add_subplot(*grid, 3)
    ax3.set_title("Average Queue Size")
    ax3.plot(queue_size["epoch"], queue_size["size"], linewidth=1)

    ax4 = fig.add_subplot(*grid, 4)
    ax4.set_title("Goodput (Tasks per Second)")
    ax4.plot(goodput["epoch"], goodput["goodput"], linewidth=1)

    ax5 = fig.add_subplot(*grid, 5)
    ax5.set_title("Task Retries")
    ax5.plot(retries["epoch"], retries["retries"], linewidth=1)

    ax6 = fig.add_subplot(*grid, 6)
    ax6.set_title("Sends per Second")
    ax6.plot(sends["epoch"], sends["sends"], linewidth=1)

    plt.show(block=True)


def main():
    experiment2()


if __name__ == "__main__":
    raise SystemExit(main())
