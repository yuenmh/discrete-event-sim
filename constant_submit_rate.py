import concurrent.futures
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from tqdm import tqdm

from des.sim import (
    Addr,
    EventLoop,
    Ref,
    RunResult,
    StateMachineBase,
    Timeout,
    handle,
    launch,
    log,
    now,
    rng,
    send,
    sleep,
    spawn,
    spawn_interface,
    this,
    wait,
    wait_timeout,
)
from des.stdlib import LaunchedStateMachine, Semaphore

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def constant_rate(wait_time: int):
    def policy(_try_num: int) -> int:
        return wait_time

    return policy


def linear_backoff(base_wait_time: int, per_try: int):
    def policy(try_num: int) -> int:
        return base_wait_time + (per_try * try_num)

    return policy


def smooth_series(series: pl.Series, avg_window: int) -> pl.Series:
    return pl.Series(
        name=series.name,
        values=np.convolve(
            series.to_numpy(),
            np.ones(avg_window) / avg_window,
        )[: len(series)],
    )


TASK_DONE = "task done"
WORKER_COMPLETED = "work_completed"
QUEUE_SIZE = "queue size"
CLIENT_SUBMIT = "client submit"
CLIENT_RETRY = "client retry"
TASK_FAILED = "task failed"


def analyze_data(result: RunResult):
    max_epoch_s = max(e.epoch for e in result.logs) // 1000 + 1

    avg_window = 10

    goodput = (
        pl.DataFrame(
            [
                (entry.data["start"] // 1000,)
                for entry in result.logs
                if entry.msg == TASK_DONE
            ],
            schema=["epoch"],
            orient="row",
        )
        .group_by("epoch", maintain_order=True)
        .agg(pl.len().alias("completed"))
        .join(
            pl.DataFrame().with_columns(pl.arange(0, max_epoch_s).alias("epoch")),
            on="epoch",
            how="right",
        )
        .fill_null(0)
        .with_columns(pl.col("completed").cum_sum().alias("completed"))
        .with_columns(
            (pl.col("completed").diff() / pl.col("epoch").diff()).alias("goodput")
        )
    )
    goodput = goodput.with_columns(
        smooth_series(goodput.get_column("goodput"), avg_window=avg_window)
    )

    queue_size = (
        pl.DataFrame(
            [
                (entry.epoch // 1000, entry.data["size"])
                for entry in result.logs
                if entry.msg == QUEUE_SIZE
            ],
            schema=["epoch", "size"],
            orient="row",
        )
        .group_by("epoch", maintain_order=True)
        .agg(pl.col("size").mean())
    )
    queue_size = queue_size.with_columns(
        smooth_series(queue_size.get_column("size"), avg_window=avg_window)
    )

    sends = (
        pl.DataFrame(
            [
                (entry.epoch // 3000)
                for entry in result.logs
                if entry.msg == CLIENT_SUBMIT
            ],
            schema=["epoch"],
            orient="row",
        )
        .group_by("epoch", maintain_order=True)
        .agg(pl.len().alias("sends"))
        .with_columns(pl.col("epoch") * 3)
    )
    sends = sends.with_columns(
        smooth_series(sends.get_column("sends"), avg_window=avg_window)
    )

    retries = (
        pl.DataFrame(
            [
                (entry.epoch // 3000)
                for entry in result.logs
                if entry.msg == CLIENT_RETRY
            ],
            schema=["epoch"],
            orient="row",
        )
        .group_by("epoch", maintain_order=True)
        .agg(pl.len().alias("retries"))
        .with_columns(pl.col("epoch") * 3)
    )
    retries = retries.join(
        pl.DataFrame().with_columns(pl.arange(0, max_epoch_s).alias("epoch")),
        on="epoch",
        how="right",
    ).fill_null(0)
    retries = retries.with_columns(
        smooth_series(retries.get_column("retries"), avg_window=avg_window)
    )

    compute_time = (
        pl.DataFrame(
            [
                (
                    entry.epoch // 1000,
                    entry.data["compute_time"],
                )
                for entry in result.logs
                if entry.msg == WORKER_COMPLETED
            ],
            schema=["epoch", "compute_time"],
            orient="row",
        )
        .group_by("epoch", maintain_order=True)
        .agg(pl.col("compute_time").mean())
    )
    compute_time = compute_time.with_columns(
        smooth_series(compute_time.get_column("compute_time"), avg_window=avg_window)
    )

    return {
        "goodput": goodput,
        "compute_time": compute_time,
        "queue_size": queue_size,
        "sends": sends,
        "retries": retries,
    }


class Queue[T](StateMachineBase):
    def __init__(self, capacity: int, log_event: str = "queue size"):
        self._capacity = capacity
        self._items = deque[T]()
        self._waiters: deque[tuple[Addr, Ref]] = deque()
        self._log_event = log_event

    def _log_size(self):
        log(self._log_event, size=len(self._items))

    @handle()
    async def try_enqueue(self, item: T, result: tuple[Addr, Ref] | None = None):
        """Attempt to enqueue an item.
        If result is provided, it sends `True` if it was enqueued, `False` if full.
        """
        if len(self._items) >= self._capacity:
            if result is not None:
                send(*result, False)
        else:
            self._items.append(item)
            while self._waiters and self._items:
                send(*self._waiters.popleft(), self._items.popleft())
            self._log_size()
            if result is not None:
                send(*result, True)

    @handle()
    async def _dequeue(self, sender: Addr, ref: Ref):
        if self._items:
            send(sender, ref, self._items.popleft())
            self._log_size()
        else:
            self._waiters.append((sender, ref))

    async def dequeue(self, timeout: int | None = None) -> T:
        """Dequeue an item, waiting up to timeout if specified.
        If timeout is provided, and no item is available within the timeout,
        raises `des.sim.Timeout`.
        """
        self._dequeue(this(), ref := Ref())
        if timeout is None:
            result, *_ = await wait(ref)
        else:
            result, *_ = await wait_timeout(timeout, ref)
        return result


def experiment2(
    work_time: int = 1000,
    work_time_rand: int = 200,
    timeout: int = 1900,
    inter_sleep: int = 4000,
    trigger_delay: int = 1_000_000,
    shared_seed: int = 0,
    local_seed: int = 100,
):
    class Worker(LaunchedStateMachine):
        def __init__(self, queue: Queue[tuple[Addr, WorkItem]]):
            self._queue = queue

        async def start(self):
            while True:
                addr, ref = await self._queue.dequeue()
                compute_time = (
                    work_time
                    - work_time_rand
                    + int(rng().expovariate(1 / work_time_rand))
                )
                await sleep(compute_time)
                log(WORKER_COMPLETED, ref=ref, compute_time=compute_time)
                send(addr, ref, hint="done")

    class BackgroundSubmit(LaunchedStateMachine):
        def __init__(
            self,
            worker_queue: Queue[tuple[Addr, WorkItem]],
            semaphore: Semaphore,
            timeout_policy: Callable[[int], int],
        ):
            self.worker_queue = worker_queue
            self.semaphore = semaphore
            self.timeout_policy = timeout_policy

        async def start(self):
            spawned_time = now()
            await self.semaphore.acquire()
            start_time = now()
            # remove stale submissions.
            # NOTE: this kind of defeats the purpose of the semaphore, so maybe
            # this should be done differently.
            if start_time - 10 > spawned_time:
                return
            num_retries = 0
            # TODO: should this be before or after acquiring the semaphore?
            log(CLIENT_SUBMIT)
            while True:
                item = WorkItem(Ref())
                self.worker_queue.try_enqueue((this(), item))
                try:
                    await wait_timeout(self.timeout_policy(num_retries), item)
                    log(
                        TASK_DONE,
                        retries=num_retries,
                        start=start_time,
                        duration=now() - start_time,
                    )
                    break
                except Timeout:
                    num_retries += 1
                if num_retries >= 6:
                    log(TASK_FAILED)
                    break
            self.semaphore.release()

    @dataclass
    class WorkItem:
        """Newtype of `Ref` to make it clear that this does not represent a simple response channel."""

        id: Ref

    class Trigger(LaunchedStateMachine):
        def __init__(self, worker_queue: Queue[tuple[Addr, WorkItem]]):
            self.worker_queue = worker_queue

        async def start(self):
            await sleep(trigger_delay)
            for _ in range(20):
                self.worker_queue.try_enqueue((this(), WorkItem(Ref())))
                await sleep(50)

    @launch
    async def main():
        worker_queue = spawn_interface(
            Queue[tuple[Addr, WorkItem]](capacity=20, log_event=QUEUE_SIZE),
            name="worker_queue",
        )
        spawn(Worker(queue=worker_queue), name="worker")
        semaphore = spawn_interface(Semaphore(max_count=4), name="semaphore")
        spawn(Trigger(worker_queue=worker_queue), name="trigger")

        while True:
            spawn(
                BackgroundSubmit(
                    worker_queue=worker_queue,
                    semaphore=semaphore,
                    timeout_policy=constant_rate(timeout),
                ),
                name="submit",
            )
            await sleep(timeout + inter_sleep)

    el = EventLoop(seed=shared_seed)
    el.spawn(Addr("main"), main)
    reseed_time = trigger_delay + 100_000
    el.run(reseed_time)

    el.seed_all(local_seed)
    return (
        analyze_data(el.run(20_000_000)),
        {
            "local_seed": local_seed,
            "reseed_time": reseed_time // 1000,
            "trigger_time": trigger_delay // 1000,
        },
    )


def plot_single_run_metrics(metrics: dict[str, pl.DataFrame]):
    goodput = metrics["goodput"]
    queue_size = metrics["queue_size"]
    sends = metrics["sends"]
    retries = metrics["retries"]

    fig = plt.figure(figsize=(18, 8))
    fig.suptitle("Metrics")
    fig.tight_layout()

    grid = (2, 2)

    ax1 = fig.add_subplot(*grid, 1)
    ax1.set_title("Completed Tasks and Goodput")
    ax1.plot(
        goodput["epoch"],
        goodput["completed"],
        label="Completed Tasks",
        color="orange",
        linewidth=1,
    )
    ax1.set_ylabel("Completed Tasks")
    ax1b = ax1.twinx()
    ax1b.plot(
        goodput["epoch"], goodput["goodput"], label="Goodput", color="teal", linewidth=1
    )
    ax1b.set_ylabel("Goodput (Tasks per UT)")
    ax1.legend(loc="upper right")
    ax1b.legend(loc="lower right")

    ax3 = fig.add_subplot(*grid, 2)
    ax3.set_title("Queue Size")
    ax3.plot(queue_size["epoch"], queue_size["size"], linewidth=1)

    ax5 = fig.add_subplot(*grid, 3)
    ax5.set_title("Task Retries and Sends per UT")
    ax5.plot(
        retries["epoch"], retries["retries"], label="Retries", color="pink", linewidth=1
    )
    ax5.set_ylabel("Retries")
    ax5b = ax5.twinx()
    ax5b.plot(
        sends["epoch"], sends["sends"], label="Sends", color="purple", linewidth=1
    )
    ax5b.set_ylabel("Sends per UT")
    ax5.legend(loc="upper right")
    ax5b.legend(loc="lower right")

    return fig


def metastable_comparison_plot():
    num_trials = 1
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                experiment2, timeout=1910, shared_seed=0, local_seed=local_seed
            )
            for local_seed in range(num_trials)
        ]
        results = [
            fut.result()
            for fut in tqdm(concurrent.futures.as_completed(futures), total=num_trials)
        ]
    results.sort(key=lambda x: x[1]["local_seed"])

    def plot_metrics(
        ax1: Axes,
        ax2: Axes,
        ax3: Axes,
        results: list[tuple[dict[str, pl.DataFrame], dict[str, Any]]],
    ):
        ax1.set_title("Number of completed tasks across trials")
        ax1.set_xlabel("time")
        ax1.set_ylabel("completed tasks")
        for result, _ in results:
            ax1.plot(
                result["goodput"]["epoch"],
                result["goodput"]["completed"],
                alpha=0.3,
                linewidth=1,
            )
        ax1.axvline(
            x=results[0][1]["trigger_time"],
            color="green",
            linestyle="--",
            label="Time of Trigger",
        )
        ax1.axvline(
            x=results[0][1]["reseed_time"],
            color="red",
            linestyle="--",
            label="Time of Seed Change",
        )
        ax1.legend(loc="lower right")

        ax2.set_title("Queue size across trials")
        ax2.set_xlabel("time")
        ax2.set_ylabel("queue size")
        for result, _ in results:
            ax2.plot(
                result["queue_size"]["epoch"],
                result["queue_size"]["size"],
                alpha=0.5,
                linewidth=1,
            )

        ax3.set_title("Compute time")
        ax3.set_xlabel("time")
        ax3.set_ylabel("compute time")
        for result, _ in results:
            ax3.plot(
                result["compute_time"]["epoch"],
                result["compute_time"]["compute_time"],
                alpha=0.5,
                linewidth=1,
            )

    fig = plt.figure(figsize=(16, 10))
    plot_metrics(
        fig.add_subplot(2, 2, 1),
        fig.add_subplot(2, 2, 2),
        fig.add_subplot(2, 2, 3),
        results,
    )
    fig.tight_layout()

    return fig


def main():
    metastable_comparison_plot().savefig("plots/constant_submit_rate.png", dpi=300)


if __name__ == "__main__":
    raise SystemExit(main())
