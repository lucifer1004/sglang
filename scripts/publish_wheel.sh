#!/usr/bin/env bash
# Upload ./_wheels/welm_*.whl (welm_sglang + welm_sglang_router) to tencent_pypi.
set -euo pipefail
set -x
shopt -s nullglob

cd "$(dirname "$0")"
cd ..

TWINE_REPO_URL="${TWINE_REPO_URL:-https://mirrors.tencent.com/repository/pypi/tencent_pypi/simple}"
WHEELS_DIR="${WHEELS_DIR:-./_wheels}"

: "${BASE_IMAGE:?BASE_IMAGE not set}"
: "${TWINE_USERNAME:?TWINE_USERNAME not set}"
: "${TWINE_PASSWORD:?TWINE_PASSWORD not set}"

files=("${WHEELS_DIR}"/welm_*.whl)
[ ${#files[@]} -gt 0 ] || { echo "ERROR: no welm_*.whl in ${WHEELS_DIR}/" >&2; exit 1; }

files_in_container=()
for f in "${files[@]}"; do
    files_in_container+=("/wheels/$(basename "$f")")
done

echo "=== uploading ${#files[@]} wheels as ${TWINE_USERNAME} ==="
printf '  %s\n' "${files[@]}"

docker run --rm \
    -v "$(cd "${WHEELS_DIR}" && pwd):/wheels:ro" \
    -e TWINE_USERNAME="${TWINE_USERNAME}" \
    -e TWINE_PASSWORD="${TWINE_PASSWORD}" \
    -e TWINE_REPO_URL="${TWINE_REPO_URL}" \
    "${BASE_IMAGE}" \
    bash -euxc '
        /envs/train/bin/python -m pip install --quiet --upgrade \
            "twine>=6.1" "pkginfo>=1.11" "packaging>=24.2"
        /envs/train/bin/python -m twine upload \
            --repository-url "${TWINE_REPO_URL}" \
            --non-interactive \
            "$@"
    ' _ "${files_in_container[@]}"

echo "✓ ${#files[@]} wheels uploaded"
