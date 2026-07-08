#!/usr/bin/env bash
# Build welm_sglang + welm_sglang_router wheels into ./_wheels/.
# Caller must export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_WELM_SGLANG=<VER>
# for the python/ wheel (setuptools-scm-based).
set -xe

cd "$(dirname "$0")"
cd ..

: "${BASE_IMAGE:?BASE_IMAGE not set}"
: "${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_WELM_SGLANG:?not set}"

WHEELS_DIR="${WHEELS_DIR:-./_wheels}"
rm -rf "${WHEELS_DIR}" && mkdir -p "${WHEELS_DIR}"

# Build welm-sglang-router (maturin, sgl-model-gateway)
docker run --rm \
    -v "$(pwd):/src" -w /src/sgl-model-gateway/bindings/python \
    "${BASE_IMAGE}" \
    bash -euxc '
        apt-get update && apt-get install -y --no-install-recommends \
            protobuf-compiler libprotobuf-dev openssl libssl-dev pkg-config patchelf
        /envs/train/bin/python -m pip install --quiet --upgrade "maturin<1.14"
        /envs/train/bin/python -m maturin build --release --out /src/'"${WHEELS_DIR}"'
    '

# Build welm-sglang (setuptools-scm, python/)
docker run --rm \
    -v "$(pwd):/src" -w /src/python \
    -e SETUPTOOLS_SCM_PRETEND_VERSION_FOR_WELM_SGLANG \
    "${BASE_IMAGE}" \
    bash -euxc '
        /envs/train/bin/python -m pip install --quiet --upgrade \
            build "setuptools<82" setuptools_scm setuptools-rust wheel
        /envs/train/bin/python -m build --wheel --outdir /src/'"${WHEELS_DIR}"'
    '

echo "=== Built wheels ==="
ls -la "${WHEELS_DIR}/"
