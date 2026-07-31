#!/usr/bin/env bash
#
# package-extension.sh
#
# Builds a Chrome Web Store-ready ZIP of the convsearch extension.
#
# The Chrome Web Store requires manifest.json at the ROOT of the ZIP, so we
# archive the *contents* of extension/ (not the extension/ folder itself).
#
# Output: dist/convsearch-extension-v<version>.zip
#
# Usage:
#   ./scripts/package-extension.sh
#
set -euo pipefail

# Resolve repo root (this script lives in scripts/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

EXT_DIR="${REPO_ROOT}/extension"
DIST_DIR="${REPO_ROOT}/dist"
MANIFEST="${EXT_DIR}/manifest.json"

if [ ! -f "${MANIFEST}" ]; then
  echo "ERROR: manifest not found at ${MANIFEST}" >&2
  exit 1
fi

# Locate a python that actually runs (some environments ship a stub that only
# prints an install prompt). A candidate is only accepted if it can execute.
PY_BIN=""
for cand in python3 python; do
  if command -v "${cand}" >/dev/null 2>&1 && "${cand}" -c "pass" >/dev/null 2>&1; then
    PY_BIN="${cand}"
    break
  fi
done

# Read the version from manifest.json. Prefer python for a robust JSON parse,
# fall back to a grep/sed extraction if python is unavailable.
VERSION=""
if [ -n "${PY_BIN}" ]; then
  VERSION="$("${PY_BIN}" -c "import json,sys;print(json.load(open(sys.argv[1]))['version'])" "${MANIFEST}")"
fi
if [ -z "${VERSION}" ]; then
  VERSION="$(grep -o '"version"[[:space:]]*:[[:space:]]*"[^"]*"' "${MANIFEST}" | head -n1 | sed 's/.*"version"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')"
fi

if [ -z "${VERSION}" ]; then
  echo "ERROR: could not read version from ${MANIFEST}" >&2
  exit 1
fi

ZIP_NAME="convsearch-extension-v${VERSION}.zip"
ZIP_PATH="${DIST_DIR}/${ZIP_NAME}"

mkdir -p "${DIST_DIR}"
rm -f "${ZIP_PATH}"

echo "Packaging convsearch extension v${VERSION}"
echo "  source: ${EXT_DIR}"
echo "  output: ${ZIP_PATH}"

if command -v zip >/dev/null 2>&1; then
  # -r recurse, -X strip extra file attributes for reproducibility.
  (
    cd "${EXT_DIR}" && \
    zip -r -X "${ZIP_PATH}" . \
      -x '*.DS_Store' \
      -x 'Thumbs.db' \
      -x '*.map'
  )
elif [ -n "${PY_BIN}" ]; then
  # Fallback: build the zip with python, excluding OS cruft and *.map.
  "${PY_BIN}" - "${EXT_DIR}" "${ZIP_PATH}" <<'PYEOF'
import os
import sys
import zipfile

ext_dir, zip_path = sys.argv[1], sys.argv[2]
EXCLUDE_NAMES = {".DS_Store", "Thumbs.db"}

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, _dirs, files in os.walk(ext_dir):
        for name in files:
            if name in EXCLUDE_NAMES or name.endswith(".map"):
                continue
            abs_path = os.path.join(root, name)
            arcname = os.path.relpath(abs_path, ext_dir).replace(os.sep, "/")
            zf.write(abs_path, arcname)
print("built with python zipfile fallback")
PYEOF
else
  echo "ERROR: neither 'zip' nor python is available to build the archive" >&2
  exit 1
fi

# Report the result: path, size, and top-level file count.
SIZE_BYTES="$(wc -c < "${ZIP_PATH}" | tr -d ' ')"
echo ""
echo "Done."
echo "  zip:   ${ZIP_PATH}"
echo "  size:  ${SIZE_BYTES} bytes"
if [ -n "${PY_BIN}" ]; then
  "${PY_BIN}" -c "import zipfile,sys;z=zipfile.ZipFile(sys.argv[1]);print('  files:',len(z.namelist()));print('  manifest at root:','manifest.json' in z.namelist())" "${ZIP_PATH}"
fi
