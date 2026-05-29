"""
Signature-level guard for FlashInfer attention backends.

FlashInfer's prefill/decode kernels do NOT support attention sinks. To prevent
the runtime from silently producing wrong outputs for sink-aware models (e.g.
WeLMv4.5) when run with `--attention-backend flashinfer` or
`--decode-attention-backend flashinfer`, the backends rely on Python's own
keyword-argument check: their ``forward_extend`` / ``forward_decode`` methods
do **not** declare ``sinks`` and do **not** accept ``**kwargs``, so any caller
passing ``sinks=...`` will fail fast with ``TypeError``.

These tests pin down that contract by inspecting the function signatures —
no GPU, no server, no wrapper state required. Two failure modes are both
caught:

  1. Someone adds ``sinks=`` as a named parameter (would silently accept it).
  2. Someone adds ``**kwargs`` back (would silently swallow it).

If either happens, the corresponding sink-aware model will once again run on
FlashInfer with broken outputs — so we lock the signature down here.
"""

import inspect
import unittest

from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.layers.attention.flashinfer_mla_backend import FlashInferMLAAttnBackend
from sglang.test.test_utils import CustomTestCase


def _signature(method):
    """Return the inspect.Signature for an instance method on a class."""
    return inspect.signature(method)


def _has_var_keyword(sig):
    """True if the signature has a ``**kwargs``-style parameter."""
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def _accepts_kwarg(sig, name):
    """True if `name` could legally be passed as a keyword argument.

    This is True when either:
      * `name` appears as a named parameter (POSITIONAL_OR_KEYWORD / KEYWORD_ONLY), or
      * the signature has a **kwargs catch-all that would absorb it.
    """
    if name in sig.parameters:
        kind = sig.parameters[name].kind
        if kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return True
    return _has_var_keyword(sig)


# (class, method_name, human-readable label) — every forward entry point on
# the FlashInfer backends that ``radix_attention`` may dispatch into.
_FORWARD_ENTRY_POINTS = [
    (FlashInferAttnBackend, "forward_extend", "FlashInferAttnBackend.forward_extend"),
    (FlashInferAttnBackend, "forward_decode", "FlashInferAttnBackend.forward_decode"),
    (
        FlashInferMLAAttnBackend,
        "forward_extend",
        "FlashInferMLAAttnBackend.forward_extend",
    ),
    (
        FlashInferMLAAttnBackend,
        "forward_decode",
        "FlashInferMLAAttnBackend.forward_decode",
    ),
]


class TestFlashInferRejectsSinksKwarg(CustomTestCase):
    """The signature alone must already refuse a ``sinks=`` keyword."""

    def test_sinks_is_not_a_named_parameter(self):
        # Belt: `sinks` must not be added as an accepted named param. If it
        # ever shows up, FlashInfer would receive sinks and silently drop
        # them inside the kernel — exactly the bug this guard prevents.
        for cls, method_name, label in _FORWARD_ENTRY_POINTS:
            with self.subTest(label=label):
                sig = _signature(getattr(cls, method_name))
                self.assertNotIn(
                    "sinks",
                    sig.parameters,
                    f"{label} must not declare a `sinks` parameter "
                    f"(FlashInfer does not support attention sinks).",
                )

    def test_no_var_keyword_swallow(self):
        # Suspenders: no `**kwargs` catch-all either, otherwise `sinks=...`
        # from `radix_attention.unified_attention_with_output` would be
        # absorbed silently and never reach a Python-level error.
        for cls, method_name, label in _FORWARD_ENTRY_POINTS:
            with self.subTest(label=label):
                sig = _signature(getattr(cls, method_name))
                self.assertFalse(
                    _has_var_keyword(sig),
                    f"{label} must not declare **kwargs — it would silently "
                    f"swallow `sinks` and let FlashInfer produce wrong outputs.",
                )

    def test_passing_sinks_would_raise_type_error(self):
        # Combined check expressed as the actual user-visible behavior:
        # binding `sinks=<something>` to the signature must fail.  This is
        # what Python does at call time, so we exercise the same path.
        for cls, method_name, label in _FORWARD_ENTRY_POINTS:
            with self.subTest(label=label):
                sig = _signature(getattr(cls, method_name))
                self.assertFalse(
                    _accepts_kwarg(sig, "sinks"),
                    f"{label} must reject a `sinks` kwarg at call time.",
                )


if __name__ == "__main__":
    unittest.main()
