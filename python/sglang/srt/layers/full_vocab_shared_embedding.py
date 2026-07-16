from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.layers.vocab_parallel_embedding import (
    VocabParallelEmbeddingShardIndices,
)


class FullVocabSharedEmbedding(nn.Module):
    tp_size = 1
    use_attn_tp_group = False
    enable_tp = False
    is_full_vocab_shared = True
    requires_vocab_reduce = False

    def __init__(
        self,
        *,
        key: str,
        num_embeddings: int,
        embedding_dim: int,
        registry=None,
    ) -> None:
        super().__init__()
        if not key:
            raise ValueError("shared embedding weight key must not be empty")
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError("shared embedding dimensions must be positive")
        if registry is None:
            from sglang.srt.model_loader.welm_shared_embedding_weights import (
                get_welm_shared_embedding_registry,
            )

            registry = get_welm_shared_embedding_registry()
        weight = registry.get(key)
        expected_shape = (num_embeddings, embedding_dim)
        if tuple(weight.shape) != expected_shape:
            raise ValueError(
                f"shared embedding {key} has shape {tuple(weight.shape)}, "
                f"expected {expected_shape}"
            )

        self.weight_key = key
        self.num_embeddings = num_embeddings
        self.org_vocab_size = num_embeddings
        self.org_vocab_size_padded = num_embeddings
        self.num_embeddings_padded = num_embeddings
        self.num_embeddings_per_partition = num_embeddings
        self.num_org_embeddings_per_partition = num_embeddings
        self.num_added_embeddings = 0
        self.num_added_embeddings_per_partition = 0
        self.embedding_dim = embedding_dim
        self.shard_indices = VocabParallelEmbeddingShardIndices(
            padded_org_vocab_start_index=0,
            padded_org_vocab_end_index=num_embeddings,
            padded_added_vocab_start_index=num_embeddings,
            padded_added_vocab_end_index=num_embeddings,
            org_vocab_start_index=0,
            org_vocab_end_index=num_embeddings,
            added_vocab_start_index=num_embeddings,
            added_vocab_end_index=num_embeddings,
        )
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.weight.output_dim = 0
        self.weight.weight_loader = self.weight_loader

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids.long(), self.weight)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        raise RuntimeError(
            f"shared WeLM host embedding weight {self.weight_key} is immutable"
        )

    def get_sharded_to_full_mapping(self):
        return None

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, key={self.weight_key!r}"
        )
