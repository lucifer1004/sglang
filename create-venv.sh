#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 env-name venv-path" >&2
    exit 2
}

if [ "$#" -ne 2 ]; then
    usage
fi

ENV_NAME="$1"
VENV_PATH="$2"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${UV:-}" ]; then
    if command -v uv >/dev/null 2>&1; then
        UV="$(command -v uv)"
    else
        UV="${HOME}/.local/bin/uv"
    fi
fi
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
PYTORCH_CUDA_INDEX_URL="${PYTORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
SGLANG_KERNEL_INDEX_URL="${SGLANG_KERNEL_INDEX_URL:-https://docs.sglang.ai/whl/cu129/}"

if [ ! -x "${UV}" ]; then
    echo "uv not found or not executable: ${UV}" >&2
    exit 1
fi

cd "${REPO_DIR}"
GIT_DESCRIBE="$(git describe --tags --always --dirty)"
SAFE_DESCRIBE="$(printf '%s' "${GIT_DESCRIBE}" | tr '/:[:space:]' '____')"
SAFE_ENV_NAME="$(printf '%s' "${ENV_NAME}" | tr '/:[:space:]' '____')"
LOG_DIR="${LOG_DIR:-${HOME}/envs/${SAFE_ENV_NAME}-${SAFE_DESCRIBE}.logs}"
mkdir -p "${LOG_DIR}"

run_logged() {
    local name="$1"
    shift
    echo
    echo "========== ${name} =========="
    "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
}

VENV_PARENT="$(dirname "${VENV_PATH}")"
mkdir -p "${VENV_PARENT}"
VENV_PATH="$(cd "${VENV_PARENT}" && pwd)/$(basename "${VENV_PATH}")"
VENV_PYTHON="${VENV_PATH}/bin/python"
SGLANG_VERSION="$(
    python3 - <<'PY' "${GIT_DESCRIBE}"
import re
import sys

describe = sys.argv[1]
match = re.fullmatch(
    r"v?(?P<base>\d+\.\d+\.\d+)(?:\.post(?P<post>\d+))?"
    r"(?:-(?P<distance>\d+)-g(?P<sha>[0-9a-f]+))?(?:-dirty)?",
    describe,
)
if not match:
    raise SystemExit(f"cannot derive a PEP 440 version from git describe: {describe}")

base = match.group("base")
post = int(match.group("post") or 0)
distance = int(match.group("distance") or 0)
sha = match.group("sha")
if distance:
    version = f"{base}.post{post + distance}"
else:
    version = f"{base}.post{post}" if post else base
if sha:
    version = f"{version}+g{sha}"
print(version)
PY
)"

echo "Repository: ${REPO_DIR}"
echo "Git describe: ${GIT_DESCRIBE}"
echo "SGLang build version: ${SGLANG_VERSION}"
echo "Environment name: ${ENV_NAME}"
echo "Venv path: ${VENV_PATH}"
echo "Logs: ${LOG_DIR}"

run_logged create-venv "${UV}" venv --relocatable --seed --clear --python "${PYTHON_VERSION}" "${VENV_PATH}"

REQ_DIR="$(mktemp -d)"
REQ_FILE="${REQ_DIR}/sglang-runtime-requirements.txt"
PYPROJECT_PATH="${REPO_DIR}/python/pyproject.toml"
PYPROJECT_BACKUP="${REQ_DIR}/pyproject.toml.orig"
cp "${PYPROJECT_PATH}" "${PYPROJECT_BACKUP}"

restore_pyproject() {
    if [ -f "${PYPROJECT_BACKUP}" ]; then
        cp "${PYPROJECT_BACKUP}" "${PYPROJECT_PATH}"
    fi
}

cleanup() {
    restore_pyproject
    rm -rf "${REQ_DIR}"
}
trap cleanup EXIT

"${VENV_PYTHON}" - <<'PY' "${PYPROJECT_PATH}"
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
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

"${VENV_PYTHON}" - <<'PY' >"${REQ_FILE}"
import pathlib
import tomllib

pyproject = pathlib.Path("python/pyproject.toml")
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

run_logged install-torch-cu128 "${UV}" pip install \
    --python "${VENV_PYTHON}" \
    --index-url "${PYTORCH_CUDA_INDEX_URL}" \
    "torch==2.11.0" \
    "torchvision==0.26.0" \
    "torchaudio==2.11.0"
run_logged install-sglang-kernel-cu129 "${UV}" pip install \
    --python "${VENV_PYTHON}" \
    --index-url "${SGLANG_KERNEL_INDEX_URL}" \
    "sglang-kernel==0.4.2.post1+cu129"
run_logged install-runtime-deps "${UV}" pip install --python "${VENV_PYTHON}" -r "${REQ_FILE}"
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG="${SGLANG_VERSION}"
run_logged install-sglang "${UV}" pip install --python "${VENV_PYTHON}" --no-deps "${REPO_DIR}/python"

restore_pyproject
run_logged pip-check "${UV}" pip check --python "${VENV_PYTHON}"
run_logged install-verify "${VENV_PYTHON}" - <<'PY'
from importlib.metadata import version

import sglang
import torch

cuda_python = version("cuda-python")
if not cuda_python.startswith("12."):
    raise SystemExit(f"cuda-python must be 12.x, got {cuda_python}")
if not str(torch.version.cuda).startswith("12."):
    raise SystemExit(f"torch CUDA runtime must be 12.x, got {torch.version.cuda}")
print(f"sglang={version('sglang')}")
print(f"torch={torch.__version__}, torch_cuda={torch.version.cuda}")
print(f"cuda-python={cuda_python}")
print(f"sglang_module={sglang.__file__}")
print(f"python={__import__('sys').executable}")
PY

echo
echo "Created relocatable venv: ${VENV_PATH}"
echo "Use with: source ${VENV_PATH}/bin/activate"
