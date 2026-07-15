#!/bin/bash

set -e

PART="patch"

for arg in "$@"; do
  case "$arg" in
    --major) PART="major" ;;
    --minor) PART="minor" ;;
    --patch) PART="patch" ;;
    *)
      echo "Unknown argument: ${arg}" >&2
      echo "Usage: $0 [--major|--minor|--patch]" >&2
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

CARGO_TOML="${ROOT_DIR}/Cargo.toml"

# Cross-repo: platform deploys model-proxy via kustomize tag here.
PLATFORM_KUSTOMIZATION="${ROOT_DIR}/../platform/k8s/model-proxy/kustomization.yaml"

CURRENT_VERSION=$(python3 - <<PY
import re
from pathlib import Path

text = Path("${CARGO_TOML}").read_text(encoding="utf-8")
m = re.search(r'(?m)^version\\s*=\\s*"(?P<v>\\d+\\.\\d+\\.\\d+)"\\s*$', text)
if not m:
    raise SystemExit(f"Could not find version in {Path('${CARGO_TOML}').name}")
print(m.group("v"))
PY
)

NEW_VERSION=$(python3 - <<PY
major, minor, patch = map(int, "${CURRENT_VERSION}".split("."))
part = "${PART}"
if part == "major":
    major, minor, patch = major + 1, 0, 0
elif part == "minor":
    minor, patch = minor + 1, 0
else:
    patch = patch + 1
print(f"{major}.{minor}.{patch}")
PY
)

echo "=> Bumping model-proxy (rust) version: ${CURRENT_VERSION} -> ${NEW_VERSION}"

python3 - <<PY
import re
from pathlib import Path

cargo = Path("${CARGO_TOML}")
new_version = "${NEW_VERSION}"
text = cargo.read_text(encoding="utf-8")
text2, n = re.subn(
    r'(?m)^version\\s*=\\s*"(\\d+\\.\\d+\\.\\d+)"\\s*$',
    f'version = "{new_version}"',
    text,
    count=1,
)
if n != 1:
    raise SystemExit(f"Expected to update exactly one version in {cargo}")
cargo.write_text(text2 + ("" if text2.endswith("\\n") else "\\n"), encoding="utf-8")
PY

python3 - <<PY
import re
from pathlib import Path

kust = Path("${PLATFORM_KUSTOMIZATION}")
new_version = "${NEW_VERSION}"
if not kust.exists():
    raise SystemExit(f"Expected platform kustomization at {kust}, but it does not exist")

text = kust.read_text(encoding="utf-8")
m = re.search(r'(?m)^(?P<prefix>\\s*newTag:\\s*).+$', text)
if not m:
    raise SystemExit(f"Could not find newTag in {kust}")
prefix = m.group("prefix")
text2, n = re.subn(r'(?m)^\\s*newTag:\\s*.+$', f"{prefix}{new_version}", text, count=1)
if n != 1:
    raise SystemExit(f"Expected to update exactly one newTag in {kust}")
kust.write_text(text2 + ("" if text2.endswith("\\n") else "\\n"), encoding="utf-8")
PY

echo "=> Updated Cargo.toml and platform k8s model-proxy newTag to version ${NEW_VERSION}"
