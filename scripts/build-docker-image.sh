#!/bin/bash

set -e

REGISTRY="me-central1-docker.pkg.dev"
IMAGE_BASE="omegacloud-platform/model-proxy-rs/model-proxy-rs"

gcloud auth configure-docker "${REGISTRY}" --quiet

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

CURRENT_VERSION=$(grep '^version' "${ROOT_DIR}/Cargo.toml" | head -1 | sed 's/.*"\(.*\)".*/\1/')
IMAGE_TAG="${CURRENT_VERSION}"
FULL_IMAGE="${REGISTRY}/${IMAGE_BASE}:${IMAGE_TAG}"
LATEST_IMAGE="${REGISTRY}/${IMAGE_BASE}:latest"

echo "=> Generating SBOM..."
(cd "${ROOT_DIR}" && trivy fs --format cyclonedx --output model-proxy-sbom.cyclonedx.json .)

echo "=> Setting up Docker buildx for multi-platform builds..."

docker buildx create --name multiplatform-builder --use || true
docker buildx use multiplatform-builder
docker buildx inspect --bootstrap

echo "=> Building and pushing ${FULL_IMAGE} ..."

docker buildx build --platform linux/amd64,linux/arm64 \
    -f "${ROOT_DIR}/Dockerfile" \
    --tag "${FULL_IMAGE}" \
    --tag "${LATEST_IMAGE}" \
    --push \
    "${ROOT_DIR}"

echo "✅ Build completed successfully!"
echo "=> ${FULL_IMAGE}"
