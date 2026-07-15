#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."

echo "=> Generating SBOM..."
trivy fs --format cyclonedx --output model-proxy-sbom.cyclonedx.json .