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
    Timeout,
    ask,
    launch,
    log,
    now,
    rng,
    self,
    send,
    sleep,
    wait_timeout,
)
from des.stdlib import Queue

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def constant_retry(wait_time: int):
    def policy(_try_num: int) -> int:
        return wait_time

    return policy


def linear_backoff_retry(base_wait_time: int, per_try: int):
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


def analyze_data(result: RunResult):
    max_epoch_s = max(e.epoch for e in result.logs) // 1000 + 1

    avg_window = 10

    goodput = (
        pl.DataFrame(
            [
                (entry.data["start"] // 1000,)
                for entry in result.logs
                if entry.msg == "done" and entry.addr.name.startswith("client")
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
                if entry.msg == "queue size"
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
                if entry.addr.name.startswith("client") and entry.msg == "submit"
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
            [(entry.epoch // 3000) for entry in result.logs if entry.msg == "retry"],
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

    return {
        "goodput": goodput,
        "queue_size": queue_size,
        "sends": sends,
        "retries": retries,
    }


def experiment2(
    work_time: int = 1000,
    work_time_rand: int = 200,
    timeout: int = 1900,
    retry_delay: int = 100,
    inter_sleep: int = 4000,
    trigger_delay: int = 1_000_000,
    seed: int = 0,
):
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
                return False
            try:
                await wait_timeout(timeout, ref)
            except Timeout:
                log("fail", ref=ref, reason="timeout")
                return False
            return True

    worker = Worker(queue_addr)

    def create_worker():
        @launch
        async def worker_process():
            while True:
                addr, ref = await queue.dequeue()
                await sleep(
                    work_time
                    - work_time_rand
                    + int(rng().expovariate(1 / work_time_rand))
                )
                send(addr, ref, hint="done")

        return worker_process

    def create_client():
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
                log(
                    "done",
                    ref=ref,
                    retries=retries,
                    start=start,
                    duration=now() - start,
                )
                await sleep(inter_sleep)

        return client

    def create_client2():
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
                log(
                    "done",
                    ref=ref,
                    retries=retries,
                    start=start,
                    duration=now() - start,
                )
                await sleep(inter_sleep)

        return client2

    def create_trigger():
        @launch
        async def trigger():
            await sleep(trigger_delay)
            for _ in range(20):
                await worker.submit(Ref(), timeout=50)

        return trigger

    el = EventLoop(seed=seed)
    el.spawn(queue_addr, lambda: Queue.create(max_size=20))
    el.spawn(Addr("worker"), create_worker)
    el.spawn(Addr("client"), create_client)
    el.spawn(Addr("client2"), create_client2)
    el.spawn(Addr("trigger"), create_trigger)
    reseed_time = trigger_delay + 100_000
    el.run(reseed_time)

    for seed in range(10):
        new_el = el.clone()
        new_el.seed_all(seed)
        yield (
            analyze_data(new_el.run(6_000_000)),
            {
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
    results = list(tqdm(experiment2(timeout=1910, seed=0), total=10))

    def plot_metrics(
        ax1: Axes,
        ax2: Axes,
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

    fig = plt.figure(figsize=(16, 5))
    plot_metrics(
        fig.add_subplot(1, 2, 1),
        fig.add_subplot(1, 2, 2),
        results,
    )
    fig.tight_layout()

    return fig


def main():
    metastable_comparison_plot().savefig("plots/same_prefix_comparison.png", dpi=300)


if __name__ == "__main__":
    raise SystemExit(main())
