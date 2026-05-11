from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

from sglang.srt.model_executor.forward_batch_info import ForwardBatch

_current_forward_batch: Optional[ForwardBatch] = None


def get_current_forward_batch() -> Optional[ForwardBatch]:
    return _current_forward_batch


@contextmanager
def set_current_forward_batch(forward_batch: ForwardBatch):
    global _current_forward_batch
    prev_forward_batch = _current_forward_batch
    _current_forward_batch = forward_batch
    try:
        yield
    finally:
        _current_forward_batch = prev_forward_batch
