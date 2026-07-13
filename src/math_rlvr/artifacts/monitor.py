"""Self-contained NVIDIA/process resource sampler."""

import csv
import os
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import psutil


class ResourceMonitor:
    FIELDS = (
        "timestamp",
        "gpu_memory_used_mb",
        "gpu_utilization_pct",
        "power_draw_w",
        "temperature_c",
        "process_rss_mb",
        "elapsed_seconds",
    )

    def __init__(self, path: Path, interval=5.0):
        self.path = path
        self.interval = interval
        self.rows = []
        self._stop = threading.Event()
        self._thread = None
        self.started = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    def start(self):
        self.started = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=False)
        self._thread.start()

    def _gpu(self):
        try:
            out = (
                subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=False,
                )
                .stdout.strip()
                .splitlines()[0]
            )
            vals = [x.strip() for x in out.split(",")]
            return [float(x) if x not in {"N/A", "[N/A]", ""} else "" for x in vals]
        except (OSError, IndexError, ValueError, subprocess.SubprocessError):
            return ["", "", "", ""]

    def sample(self):
        gpu = self._gpu()
        elapsed = time.monotonic() - self.started
        row = dict(
            zip(
                self.FIELDS,
                [
                    datetime.now(UTC).isoformat(),
                    *gpu,
                    psutil.Process(os.getpid()).memory_info().rss / 1048576,
                    elapsed,
                ],
                strict=True,
            )
        )
        self.rows.append(row)
        self._flush()

    def _flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(f".{self.path.name}.tmp")
        with temp.open("w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=self.FIELDS)
            w.writeheader()
            w.writerows(self.rows)
        os.replace(temp, self.path)

    def _run(self):
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval + 4)
        if not self.rows:
            self.sample()

    def summary(self, price=8.88):
        def nums(key):
            return [float(row[key]) for row in self.rows if row[key] != ""]

        elapsed = max(nums("elapsed_seconds"), default=0.0)
        util = nums("gpu_utilization_pct")
        hours = elapsed / 3600
        return {
            "peak_vram_mb": max(nums("gpu_memory_used_mb"), default=None),
            "mean_gpu_utilization": sum(util) / len(util) if util else None,
            "gpu_hours": hours,
            "estimated_cost_cny": hours * price,
        }
