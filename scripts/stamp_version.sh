#!/usr/bin/env bash
# Inject <version> for both sglang wheels:
# - sgl-model-gateway/bindings/python/pyproject.toml: sed the static version
# - python/ (welm-sglang, setuptools-scm): NOT stamped here — caller must
#   export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_WELM_SGLANG=<VER>
set -euo pipefail
VER="${1:?usage: stamp_version.sh <version>}"

cd "$(dirname "$0")"
cd ..

F="sgl-model-gateway/bindings/python/pyproject.toml"
sed -i.bak -E "/^\\[project\\]/,/^\\[/{ s/^version = \"[^\"]*\"/version = \"${VER}\"/; }" "$F"
rm -f "${F}.bak"
grep -qF "version = \"${VER}\"" "$F" || { echo "FAILED to stamp version in $F" >&2; exit 1; }
echo "=== stamped version=${VER} into ${F} ==="
