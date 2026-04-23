from __future__ import annotations

from torch import nn


class CustomOp(nn.Module):
    """Compatibility shim for older SGLang custom-op modules.

    The WeLM kernels in this branch only need the lightweight device-dispatch
    behavior that used to live at ``sglang.srt.custom_op.CustomOp``.
    """

    def forward(self, *args, **kwargs):
        device_type = None
        for arg in args:
            if hasattr(arg, "device"):
                device_type = arg.device.type
                break
        if device_type == "cuda" and hasattr(self, "forward_cuda"):
            return self.forward_cuda(*args, **kwargs)
        if device_type == "xpu" and hasattr(self, "forward_xpu"):
            return self.forward_xpu(*args, **kwargs)
        if hasattr(self, "forward_native"):
            return self.forward_native(*args, **kwargs)
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement a compatible forward path"
        )
