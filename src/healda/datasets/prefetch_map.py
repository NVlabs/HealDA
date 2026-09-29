# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Async data loader that processes batches in a background thread with a separate CUDA stream.
"""

import queue
import threading
import torch
from typing import Callable, Iterable, Any, Optional
from torch.utils.data import DataLoader
import dataclasses


class _Done:
    pass


class _PrefetchIterator:
    """
    Wraps a PyTorch DataLoader to process batches asynchronously in a background thread.

    The background thread records a CUDA event after processing; the consumer stream
    waits on that event without blocking the host.
    """

    def __init__(
        self,
        dataloader: Iterable,
        transform: Callable[[Any], Any],
        queue_size: int = 2,
        cuda_stream: Optional[torch.cuda.Stream] = None,
    ):
        """
        Args:
            dataloader: The PyTorch DataLoader to wrap
            transform: Function to apply to each batch (batch -> batch)
            queue_size: Maximum size of the processing queue (default: 2)
            cuda_stream: CUDA stream to use for background processing (creates new if None)
        """
        self.dataloader = dataloader
        self.transform = transform
        self.queue_size = queue_size
        self.cuda_stream = cuda_stream or torch.cuda.Stream()

        # Threading components
        self.queue = queue.Queue(maxsize=queue_size)
        self.thread = None
        self.stop_event = threading.Event()

        # Iterator state
        self.dataloader_iter = None
        self._started = False

    def _worker(self):
        """Background worker that processes batches."""
        try:
            while not self.stop_event.is_set():
                try:
                    # Get next batch from dataloader
                    batch = next(self.dataloader_iter)
                except StopIteration:
                    # No more data, put sentinel and break
                    self.queue.put((_Done, None, None))
                    break

                # Process batch in background CUDA stream
                with torch.cuda.stream(self.cuda_stream):
                    processed_batch = self.transform(batch)
                    ready = torch.cuda.Event()
                    ready.record()

                self.queue.put((processed_batch, None, ready))

        except Exception as e:
            self.queue.put((None, e, None))

    def _start(self):
        """Start the background processing thread."""
        if self._started:
            return

        self.dataloader_iter = iter(self.dataloader)
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        self._started = True

    def __len__(self):
        return len(self.dataloader)

    def _stop(self):
        """Stop the background processing thread."""
        if not self._started:
            return

        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self._started = False

    def __iter__(self):
        """Start background processing and return iterator."""
        self._start()
        return self

    @staticmethod
    def _record_stream(x, stream):
        if isinstance(x, torch.Tensor):
            x.record_stream(stream)
        elif isinstance(x, list):
            for item in x:
                _PrefetchIterator._record_stream(item, stream)
        elif isinstance(x, dict):
            for item in x.values():
                _PrefetchIterator._record_stream(item, stream)
        elif dataclasses.is_dataclass(x):
            for field in dataclasses.fields(x):
                _PrefetchIterator._record_stream(getattr(x, field.name), stream)

    def __next__(self):
        """Get next processed batch."""
        if not self._started:
            raise RuntimeError("Iterator not started. Call __iter__ first.")

        # Get processed batch from queue
        try:
            batch, error, ready = self.queue.get()
        except queue.Empty:
            raise RuntimeError("Timeout waiting for processed batch")

        if error is not None:
            raise error

        # Check for end of data
        if batch is _Done:
            self._stop()
            raise StopIteration

        torch.cuda.current_stream().wait_event(ready)

        # By default, the caching allocator is aware of only the stream where a tensor was allocated.
        # Tell the allocator the consumer stream also owns these tensors, so their
        # memory isn't recycled while the consumer stream still is using them.
        self._record_stream(batch, torch.cuda.current_stream())

        return batch

    def __del__(self):
        """Cleanup on deletion."""
        self._stop()


def prefetch_map(
    dataloader: DataLoader,
    transform: Callable[[Any], Any],
    queue_size: int = 2,
    cuda_stream: Optional[torch.cuda.Stream] = None,
) -> _PrefetchIterator:
    """
    Create an async data loader that processes batches in a background thread.

    Args:
        dataloader: The PyTorch DataLoader to wrap
        transform: Function to apply to each batch (batch -> batch)
        queue_size: Maximum size of the processing queue (default: 2)
        cuda_stream: CUDA stream to use for background processing (creates new if None)

    Returns:
        AsyncDataLoader that can be iterated over like a regular DataLoader

    Example:
        >>> def move_to_gpu(batch):
        ...     return {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        >>>
        >>> async_loader = async_map(dataloader, move_to_gpu)
        >>> for batch in async_loader:
        ...     # batch is already on GPU and processed
        ...     pass
    """
    return _PrefetchIterator(dataloader, transform, queue_size, cuda_stream)
