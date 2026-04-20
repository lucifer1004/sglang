#!/usr/bin/env python3

import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.models import welm_perf_opt as mod


class DummyOEContext:
    def __init__(self, gram2, gram3):
        self.grams = {2: gram2, 3: gram3}

    def get_gram(self, n: int):
        return self.grams.get(n)


def bench(fn, warmup=20, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iters


def bytes_per_token(num_outputs: int, output_dtype: torch.dtype) -> int:
    input_bytes = 8 * 3  # input_ids, gram2, gram3 are int64 in runtime dumps
    output_bytes = torch.tensor([], dtype=output_dtype).element_size() * num_outputs
    return input_bytes + output_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", type=Path)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    payload = torch.load(args.dump, map_location="cpu")
    device = "cuda"
    input_ids = payload["input_ids"].to(device)
    gram2 = payload["gram2"].to(device) if payload["gram2"] is not None else None
    gram3 = payload["gram3"].to(device) if payload["gram3"] is not None else None
    oe_vocab_sizes = payload["oe_vocab_sizes"]
    vocab_size = int(payload["vocab_size"])

    ctx = DummyOEContext(gram2, gram3)

    # Use lightweight fake modules only for generic helper shape metadata.
    class Fake:
        def __init__(self):
            self.shard_indices = SimpleNamespace(
                org_vocab_start_index=0,
                org_vocab_end_index=2**31 - 1,
            )

    fake_modules = [Fake() for _ in oe_vocab_sizes]

    spec = lambda: mod._compute_welm_oe_hashed_inputs_specialized_2233(
        input_ids=input_ids,
        oe_context=ctx,
        oe_vocab_sizes=oe_vocab_sizes,
        vocab_size=vocab_size,
    )
    gen = lambda: mod._compute_welm_oe_hashed_inputs_fused(
        input_ids=input_ids,
        oe_context=ctx,
        oe_grams=[2, 2, 3, 3],
        oe_vocab_sizes=oe_vocab_sizes,
        vocab_size=vocab_size,
        oe_embed_modules=fake_modules,
        use_triton_preprocess=False,
    )

    spec()
    gen()
    torch.cuda.synchronize()

    spec_ms = bench(spec, iters=args.iters)
    gen_ms = bench(gen, iters=args.iters)

    n = input_ids.numel()
    bytes_specialized = n * bytes_per_token(len(oe_vocab_sizes), torch.int32)
    bytes_generic = n * bytes_per_token(len(oe_vocab_sizes), torch.int64)
    spec_bw = bytes_specialized / (spec_ms / 1000.0) / 1e12
    gen_bw = bytes_generic / (gen_ms / 1000.0) / 1e12

    print(f"dump={args.dump}")
    print(f"num_tokens={n}")
    print(f"specialized_ms={spec_ms:.4f}")
    print(f"generic_ms={gen_ms:.4f}")
    print(f"specialized_bw_TBps={spec_bw:.4f}")
    print(f"generic_bw_TBps={gen_bw:.4f}")


if __name__ == "__main__":
    main()
