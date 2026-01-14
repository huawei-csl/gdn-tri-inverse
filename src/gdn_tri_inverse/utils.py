#!/usr/bin/env python
# coding: utf-8
 
import torch
import torch.nn.functional as F
import torch_npu
 
import os
import types
import typing
from dataclasses import dataclass
import logging
 
_DEFAULT_MAX_CACHE_SIZE = 256 * 1024 * 1024
logger = logging.getLogger(__name__)
 
 
@dataclass
class Device:
    module: types.ModuleType
    name: str
 
    def sync(self) -> None:
        self.module.synchronize()
 
    def event(self) -> "typing.Self.module.Event":
        return self.module.Event(enable_timing=True)
 
    def id(self) -> int:
        if ":" in self.name:
            try:
                return int(self.name.split(":")[1])
            except Exception as e:
                logger.error(f"Invalid input device string. Got {self.name}")
                return 0
        else:
            return 0
 
    def device_type(self) -> str:
        return self.name.split(":")[0]
 
 
def run_benchmark(
    device: Device,
    fn: typing.Callable,
    warmup_iters: int = 1,
    benchmark_iters: int = 5,
):
    """
    Benchmark a given function with warmup.
 
    Args:
        device: Device to run benchmark on.
        fn: Function to benchmark.
        warmup_iters: Number of warmup runs.
        benchmark_iters: Number of benchmark runs.
 
    Returns:
        Average time in microseconds.
    """
    torch.npu.set_device(device.id())
 
    start_events = [device.event() for _ in range(benchmark_iters)]
    end_events = [device.event() for _ in range(benchmark_iters)]
 
    device.sync()
    for i in range(warmup_iters):
        logger.info(f"Warmup iteration: {i}")
        fn()
 
    device.sync()
 
    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2 cache
    # doesn't contain any input data before the run
    # Copied from https://github.com/triton-lang/triton/blob/v2.1.0/python/triton/testing.py#L110
    cache_size = _DEFAULT_MAX_CACHE_SIZE
    cache = torch.ones(cache_size, dtype=torch.int8, device=device.name)
 
    for i in range(benchmark_iters):
        logger.info(f"Benchmarking iteration {i}")
        cache.zero_()
        device.sync()
        start_events[i].record()
        fn()
        end_events[i].record()
        device.sync()
        elapsed_time_ms = int(start_events[i].elapsed_time(end_events[i]))
        logger.info(f"Elapsed time: {elapsed_time_ms:,} ms")
        yield elapsed_time_ms