# syntax=docker/dockerfile:1.7
FROM --platform=$BUILDPLATFORM golang:1.25.3-bookworm AS go-builder

WORKDIR /go-build

ARG TARGETOS=linux
ARG TARGETARCH=amd64

COPY 3rdparty/sr/go.mod /go-build/sr/go.mod
COPY 3rdparty/jfs-fast-get/go.mod 3rdparty/jfs-fast-get/go.sum /go-build/jfs-fast-get/
COPY 3rdparty/jfs-fast-get/third_party/juicefs/go.mod 3rdparty/jfs-fast-get/third_party/juicefs/go.sum /go-build/jfs-fast-get/third_party/juicefs/
COPY 3rdparty/jfs-fast-get/third_party/gosigar/go.mod /go-build/jfs-fast-get/third_party/gosigar/go.mod

RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    cd /go-build/sr && go mod download && \
    cd /go-build/jfs-fast-get && go mod download

COPY 3rdparty/sr /go-build/sr
COPY 3rdparty/jfs-fast-get /go-build/jfs-fast-get

RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    cd /go-build/sr && \
    CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} \
        go build -trimpath -ldflags="-s -w" -o /out/sr . && \
    cd /go-build/jfs-fast-get && \
    CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} \
        go build -trimpath -ldflags="-s -w" -o /out/jfs-fast-get ./cmd/jfs-fast-get

FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

ARG PYTHON_VERSION=3.12
ARG VENV_PATH=/envs/venv
ARG TCCL_VERSION=2.28
ARG PYTORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG SGLANG_KERNEL_INDEX_URL=https://docs.sglang.ai/whl/cu129/
ARG SGLANG_VERSION=0.0.0.dev0

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=${VENV_PATH} \
    PATH="${VENV_PATH}/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/tccl/lib:/usr/local/cuda/lib64" \
    NCCL_IB_TC=160 \
    NCCL_SOCKET_IFNAME=bond1 \
    NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7,mlx5_bond_8 \
    NCCL_IB_GID_INDEX=3 \
    NCCL_IB_QPS_PER_CONNECTION=4 \
    NCCL_IB_TIMEOUT=22 \
    NCCL_IB_SL=3 \
    NCCL_IB_DISABLE=0 \
    GLOO_SOCKET_IFNAME=bond1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        curl \
        git \
        libibverbs-dev \
        libnuma1 \
        protobuf-compiler \
        python3 \
        python3-dev \
        unzip \
        wget \
        xz-utils \
        zstd \
        bzip2 \
        dstat \
        htop && \
    rm -rf /var/lib/apt/lists/*

COPY --from=go-builder /out/sr /usr/local/bin/sr
COPY --from=go-builder /out/jfs-fast-get /usr/local/bin/jfs-fast-get

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path --profile minimal
ENV PATH="/root/.local/bin:/root/.cargo/bin:${PATH}"

# Inline create-venv.sh's venv creation and managed-Python embedding so this
# Dockerfile does not depend on a prebuilt venv.tar or an external script.
RUN --mount=type=cache,target=/root/.cache/uv <<'BASH'
set -euo pipefail
curl -LsSf https://astral.sh/uv/install.sh | sh
/root/.local/bin/uv venv \
        --relocatable \
        --seed \
        --clear \
        --managed-python \
        --python "${PYTHON_VERSION}" \
        "${VENV_PATH}"
venv_python="${VENV_PATH}/bin/python"
base_prefix="$("${venv_python}" -c 'import sys; print(sys.base_prefix)')"
python_exe_name="$("${venv_python}" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
rm -rf "${VENV_PATH}/.python"
mkdir -p "${VENV_PATH}/.python"
cp -a "${base_prefix}/." "${VENV_PATH}/.python/"
"${venv_python}" - <<'PY' "${VENV_PATH}/.python"
import pathlib
import sys

embedded_prefix = pathlib.Path(sys.argv[1]).resolve()
for path in (embedded_prefix / "lib").glob("python*/_sysconfigdata*.py"):
    text = path.read_text()
    marker = "build_time_vars = {"
    if "_sglang_embedded_prefix" in text or marker not in text:
        continue
    text = text.replace(
        marker,
        "import pathlib as _sglang_pathlib\n"
        "_sglang_embedded_prefix = _sglang_pathlib.Path(__file__).resolve().parents[2]\n"
        + marker,
        1,
    )
    text += (
        "\n_sglang_include = str(_sglang_embedded_prefix / 'include' / "
        "('python' + build_time_vars.get('LDVERSION', build_time_vars.get('VERSION', ''))))\n"
        "for _sglang_key in ('INCLUDEPY', 'CONFINCLUDEPY'):\n"
        "    if _sglang_key in build_time_vars:\n"
        "        build_time_vars[_sglang_key] = _sglang_include\n"
    )
    path.write_text(text)
PY
ln -sfn "../.python/bin/${python_exe_name}" "${VENV_PATH}/bin/python"
ln -sfn python "${VENV_PATH}/bin/python3"
ln -sfn python "${VENV_PATH}/bin/${python_exe_name}"
BASH

WORKDIR /sgl-workspace

COPY python/pyproject.toml /sgl-workspace/python/pyproject.toml

RUN "${VENV_PATH}/bin/python" - <<'PY'
import pathlib

path = pathlib.Path("/sgl-workspace/python/pyproject.toml")
text = path.read_text()
replacements = {
    '"cuda-python>=13.0",': '"cuda-python>=12,<13",',
    '"sglang-kernel==0.4.2",': '"sglang-kernel==0.4.2.post1+cu129",',
    '"nvidia-cutlass-dsl==4.4.2",': '"nvidia-cutlass-dsl==4.5.0",',
}
for old, new in replacements.items():
    if old not in text:
        raise SystemExit(f"expected dependency line not found in {path}: {old}")
    text = text.replace(old, new, 1)
path.write_text(text)
PY

# Install torch first; this is the largest and most stable dependency layer.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python "${VENV_PATH}/bin/python" \
        --index-url "${PYTORCH_CUDA_INDEX_URL}" \
        "torch==2.11.0" \
        "torchvision==0.26.0" \
        "torchaudio==2.11.0"

# TCCL must be installed after torch is importable, before the rest of Python
# dependencies pull in or compile against communication libraries.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python "${VENV_PATH}/bin/python" requests

RUN mkdir -p /src/tccl && cd /src/tccl && \
    wget -O tccl_install.py \
        https://mirrors.tencent.com/repository/generic/TCCL/NV/tccl_install.py && \
    "${VENV_PATH}/bin/python" tccl_install.py -v "${TCCL_VERSION}" --install_mode compile_install --skip_check --clean 2>&1 | tee /tmp/tccl_install.log && \
    ! grep -q "TCCL Install.*ERROR" /tmp/tccl_install.log && \
    grep -q "Successfully installed" /tmp/tccl_install.log && \
    test -s /usr/local/tccl/lib/libtccl.so.2 && \
    test -s /usr/local/tccl/lib/plugin/libnccl-profiler-inspector.so && \
    test -s /usr/local/tccl/lib/plugin/libnccl-tuner-astralNet.so && \
    cd / && rm -rf /src/tccl

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python "${VENV_PATH}/bin/python" \
        --index-url "${SGLANG_KERNEL_INDEX_URL}" \
        "sglang-kernel==0.4.2.post1+cu129"

RUN "${VENV_PATH}/bin/python" - <<'PY' > /tmp/sglang-runtime-requirements.txt
import pathlib
import tomllib

pyproject = pathlib.Path("/sgl-workspace/python/pyproject.toml")
data = tomllib.loads(pyproject.read_text())
for dep in data["project"]["dependencies"]:
    dep_lower = dep.strip().lower()
    if dep_lower.startswith("cuda-python"):
        print("cuda-python>=12,<13")
    elif dep_lower == "numpy":
        print("numpy==2.2.6")
    elif dep_lower.startswith(("torch==", "torchaudio==", "torchvision")):
        continue
    elif dep_lower.startswith("sglang-kernel=="):
        continue
    else:
        print(dep)
PY

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python "${VENV_PATH}/bin/python" -r /tmp/sglang-runtime-requirements.txt

COPY README.md LICENSE /sgl-workspace/
COPY python /sgl-workspace/python
COPY proto /sgl-workspace/proto
COPY rust/sglang-grpc /sgl-workspace/rust/sglang-grpc

RUN "${VENV_PATH}/bin/python" - <<'PY'
import pathlib

path = pathlib.Path("/sgl-workspace/python/pyproject.toml")
text = path.read_text()
replacements = {
    '"cuda-python>=13.0",': '"cuda-python>=12,<13",',
    '"sglang-kernel==0.4.2",': '"sglang-kernel==0.4.2.post1+cu129",',
    '"nvidia-cutlass-dsl==4.4.2",': '"nvidia-cutlass-dsl==4.5.0",',
}
for old, new in replacements.items():
    if old not in text:
        raise SystemExit(f"expected dependency line not found in {path}: {old}")
    text = text.replace(old, new, 1)
path.write_text(text)
PY

RUN --mount=type=cache,target=/root/.cache/uv \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG="${SGLANG_VERSION}" \
    uv pip install --python "${VENV_PATH}/bin/python" --no-deps /sgl-workspace/python && \
    uv pip check --python "${VENV_PATH}/bin/python" && \
    "${VENV_PATH}/bin/python" - <<'PY'
from importlib.metadata import version
import pathlib
import sys
import sysconfig

import sglang
import torch

include_dir = pathlib.Path(sysconfig.get_config_var("INCLUDEPY"))
python_h = include_dir / "Python.h"
if not python_h.is_file():
    raise SystemExit(f"Python.h not found at {python_h}")
cuda_python = version("cuda-python")
if not cuda_python.startswith("12."):
    raise SystemExit(f"cuda-python must be 12.x, got {cuda_python}")
if not str(torch.version.cuda).startswith("12."):
    raise SystemExit(f"torch CUDA runtime must be 12.x, got {torch.version.cuda}")
print(f"python={sys.executable}")
print(f"python_include={include_dir}")
print(f"sglang={version('sglang')}")
print(f"torch={torch.__version__}, torch_cuda={torch.version.cuda}")
print(f"cuda-python={cuda_python}")
print(f"sglang_module={sglang.__file__}")
PY

WORKDIR /sgl-workspace
CMD ["/bin/bash"]
