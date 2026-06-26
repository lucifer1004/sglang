from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.managers.tp_worker import TpModelWorker


class BaseDraftWorker(ABC):
    @abstractmethod
    def draft():
        pass

    @abstractmethod
    def draft_extend():
        pass


class BaseSpecWorker(ABC):
    @property
    @abstractmethod
    def target_worker(self) -> TpModelWorker:
        pass

    @property
    @abstractmethod
    def draft_worker(self) -> BaseDraftWorker:
        pass

    @abstractmethod
    def clear_cache_pool(self):
        # TODO: move this abstract method to BaseTpWorker and call through self.model_runner
        pass

    def on_verify_complete_cpu(self, num_correct_drafts_per_req: list[int]) -> None:
        """Hook called after verify finishes and accept counts are on CPU.

        Default no-op. Adaptive-aware workers override this to feed the
        controller without forcing a GPU→CPU sync in the worker hot path.
        """
        pass

    def load_weights_from_distributed(self, received):
        """Load already-received (name, tensor) pairs into the draft model.

        Called by the scheduler mixin after the target received the weights
        once via NCCL. Subclasses that own draft weights (e.g. EAGLEWorkerV2,
        MultiLayerEagleWorkerV2) override this to reach their inner draft
        runner. The default raises so missing overrides are caught early.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement "
            "load_weights_from_distributed; draft weights cannot be updated "
            "via the distributed path."
        )
