"""Latency, memory, and energy-proxy profiling utilities."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch


@dataclass
class ProfileResult:
    """Single-forward-pass hardware profile summary."""

    latency_ms: float
    peak_memory_mb: float
    average_power_w: Optional[float]
    energy_joules: Optional[float]


class NVMLPowerSampler:
    """Background NVML power sampler used to approximate inference energy."""

    def __init__(self, device_index: int, interval_seconds: float = 0.05) -> None:
        self.device_index = device_index
        self.interval_seconds = interval_seconds
        self.samples: List[float] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._pynvml = None
        self._handle = None

        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception:
            self._pynvml = None
            self._handle = None

    @property
    def available(self) -> bool:
        """Whether NVML sampling is usable in the current environment."""

        return self._pynvml is not None and self._handle is not None

    def start(self) -> None:
        """Start background power sampling."""

        if not self.available:
            return
        self.samples = []
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Optional[float]:
        """Stop sampling and return the average power in Watts."""

        if not self.available:
            return None
        self._running = False
        if self._thread is not None:
            self._thread.join()
        if not self.samples:
            return None
        return sum(self.samples) / len(self.samples)

    def _sample_loop(self) -> None:
        """Continuously sample device power while profiling is active."""

        assert self._pynvml is not None
        while self._running:
            try:
                milliwatts = self._pynvml.nvmlDeviceGetPowerUsage(self._handle)
                self.samples.append(milliwatts / 1000.0)
            except Exception:
                break
            time.sleep(self.interval_seconds)


def profile_callable(
    fn: Callable[[], None],
    device: torch.device,
    repeats: int = 10,
    warmup: int = 2,
) -> ProfileResult:
    """Profile latency, peak memory, and approximate energy for a callable."""

    for _ in range(max(0, warmup)):
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        power_sampler = NVMLPowerSampler(device_index)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        power_sampler.start()
        start_event.record()
        for _ in range(repeats):
            fn()
        end_event.record()
        torch.cuda.synchronize(device)
        average_power = power_sampler.stop()

        total_latency_ms = start_event.elapsed_time(end_event)
        latency_ms = total_latency_ms / max(repeats, 1)
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        energy_joules = None if average_power is None else average_power * (latency_ms / 1000.0)
        return ProfileResult(
            latency_ms=latency_ms,
            peak_memory_mb=peak_memory_mb,
            average_power_w=average_power,
            energy_joules=energy_joules,
        )

    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    latency_ms = ((time.perf_counter() - start) / max(repeats, 1)) * 1000.0
    return ProfileResult(
        latency_ms=latency_ms,
        peak_memory_mb=0.0,
        average_power_w=None,
        energy_joules=None,
    )
