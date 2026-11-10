#!/usr/bin/env bash

set -euo pipefail

DEFAULT_REPO="m0n0x41d/anthropic-proxy-rs"
DEFAULT_REPO_URL="https://github.com/$DEFAULT_REPO"
DEFAULT_PACKAGE_NAME="anthropic-proxy"

usage() {
    cat <<'EOF'
Usage: install.sh [options]

Install anthropic-proxy. Tries downloading a pre-built binary first,
falls back to cargo install from git.

Options:
  --repo <url>      Git repository URL
  --root <dir>      Install root (binary goes to <root>/bin/)
  --branch <name>   Install from a git branch (cargo only)
  --tag <name>      Install a specific version tag
  --rev <sha>       Install from a git revision (cargo only)
  --force           Reinstall even if already installed
  --cargo           Skip binary download, use cargo install
  -h, --help        Show this help

Environment overrides:
  ANTHROPIC_PROXY_INSTALL_REPO
  ANTHROPIC_PROXY_INSTALL_ROOT
  ANTHROPIC_PROXY_INSTALL_BRANCH
  ANTHROPIC_PROXY_INSTALL_TAG
  ANTHROPIC_PROXY_INSTALL_REV
  ANTHROPIC_PROXY_INSTALL_FORCE=1
EOF
}

need_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        return 1
    fi
}

detect_target() {
    local arch os
    arch="$(uname -m)"
    os="$(uname -s)"

    case "$os" in
        Linux)  os="unknown-linux-gnu" ;;
        Darwin) os="apple-darwin" ;;
        *)      echo "Unsupported OS: $os" >&2; return 1 ;;
    esac

    case "$arch" in
        x86_64|amd64)  arch="x86_64" ;;
        aarch64|arm64) arch="aarch64" ;;
        *)             echo "Unsupported architecture: $arch" >&2; return 1 ;;
    esac

    echo "${arch}-${os}"
}

try_binary_install() {
    local tag="$1" target="$2" install_dir="$3"

    if ! need_cmd curl && ! need_cmd wget; then
        echo "Neither curl nor wget found, skipping binary download" >&2
        return 1
    fi

    local repo="${ANTHROPIC_PROXY_INSTALL_REPO:-$DEFAULT_REPO}"
    local base_url="https://github.com/${repo}/releases"

    if [[ -z "$tag" ]]; then
        tag="$(curl -fsSL -o /dev/null -w '%{url_effective}' "${base_url}/latest" 2>/dev/null \
               | grep -oE 'v[0-9]+\.[0-9]+\.[0-9]+' || true)"
        if [[ -z "$tag" ]]; then
            echo "Could not detect latest release" >&2
            return 1
        fi
    fi

    local archive_name="${DEFAULT_PACKAGE_NAME}-${target}.tar.gz"
    local url="${base_url}/download/${tag}/${archive_name}"
    echo "Downloading ${DEFAULT_PACKAGE_NAME} ${tag} for ${target}..."

    local tmpdir
    tmpdir="$(mktemp -d)"
    trap 'rm -rf "$tmpdir"' EXIT

    if need_cmd curl; then
        curl -fsSL "$url" -o "${tmpdir}/${archive_name}" || return 1
    else
        wget -q "$url" -O "${tmpdir}/${archive_name}" || return 1
    fi

    # --- checksum verification ---
    local sums_url="${base_url}/download/${tag}/SHA256SUMS"
    if need_cmd curl; then
        curl -fsSL "$sums_url" -o "${tmpdir}/SHA256SUMS" 2>/dev/null || true
    else
        wget -q "$sums_url" -O "${tmpdir}/SHA256SUMS" 2>/dev/null || true
    fi

    if [[ -f "${tmpdir}/SHA256SUMS" ]]; then
        local expected
        expected="$(grep "${archive_name}" "${tmpdir}/SHA256SUMS" | awk '{print $1}')"
        if [[ -n "$expected" ]]; then
            local actual
            if command -v sha256sum >/dev/null 2>&1; then
                actual="$(sha256sum "${tmpdir}/${archive_name}" | awk '{print $1}')"
            elif command -v shasum >/dev/null 2>&1; then
                actual="$(shasum -a 256 "${tmpdir}/${archive_name}" | awk '{print $1}')"
            else
                echo "WARNING: no sha256sum/shasum available, skipping checksum verification" >&2
                actual=""
            fi

            if [[ -n "$actual" ]]; then
                if [[ "$actual" != "$expected" ]]; then
                    echo "ERROR: checksum mismatch for ${archive_name}" >&2
                    echo "  expected: ${expected}" >&2
                    echo "  actual:   ${actual}" >&2
                    return 1
                fi
                echo "Checksum verified ✓"
            fi
        else
            echo "WARNING: ${archive_name} not found in SHA256SUMS, skipping verification" >&2
        fi
    else
        echo "WARNING: SHA256SUMS not available for ${tag}, skipping verification" >&2
    fi
    # --- end checksum verification ---

    tar xzf "${tmpdir}/${archive_name}" -C "$tmpdir"
    mkdir -p "$install_dir"
    mv "${tmpdir}/${DEFAULT_PACKAGE_NAME}" "${install_dir}/${DEFAULT_PACKAGE_NAME}"
    chmod +x "${install_dir}/${DEFAULT_PACKAGE_NAME}"
    echo "Installed ${DEFAULT_PACKAGE_NAME} ${tag} to ${install_dir}/${DEFAULT_PACKAGE_NAME}"
}

cargo_install() {
    need_cmd cargo || { echo "cargo not found. Install Rust: https://rustup.rs" >&2; exit 127; }

    local repo_url="${ANTHROPIC_PROXY_INSTALL_REPO:-$DEFAULT_REPO_URL}"
    local cmd=(cargo install --locked --git "$repo_url")

    [[ -n "$install_root" ]] && cmd+=(--root "$install_root")
    [[ -n "$git_branch" ]] && cmd+=(--branch "$git_branch")
    [[ -n "$git_tag" ]] && cmd+=(--tag "$git_tag")
    [[ -n "$git_rev" ]] && cmd+=(--rev "$git_rev")
    [[ "$force_install" == "1" ]] && cmd+=(--force)

    cmd+=("$DEFAULT_PACKAGE_NAME")

    printf 'Running:'
    printf ' %q' "${cmd[@]}"
    printf '\n'

    "${cmd[@]}"

    local binary_path
    if [[ -n "$install_root" ]]; then
        binary_path="$install_root/bin/$DEFAULT_PACKAGE_NAME"
    else
        binary_path="${CARGO_HOME:-$HOME/.cargo}/bin/$DEFAULT_PACKAGE_NAME"
    fi

    echo "Installed $DEFAULT_PACKAGE_NAME to $binary_path"
}

repo_url="${ANTHROPIC_PROXY_INSTALL_REPO:-$DEFAULT_REPO_URL}"
install_root="${ANTHROPIC_PROXY_INSTALL_ROOT:-}"
git_branch="${ANTHROPIC_PROXY_INSTALL_BRANCH:-}"
git_tag="${ANTHROPIC_PROXY_INSTALL_TAG:-}"
git_rev="${ANTHROPIC_PROXY_INSTALL_REV:-}"
force_install="${ANTHROPIC_PROXY_INSTALL_FORCE:-0}"
cargo_only=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)    repo_url="${2:?missing value for --repo}"; shift 2 ;;
        --root)    install_root="${2:?missing value for --root}"; shift 2 ;;
        --branch)  git_branch="${2:?missing value for --branch}"; shift 2 ;;
        --tag)     git_tag="${2:?missing value for --tag}"; shift 2 ;;
        --rev)     git_rev="${2:?missing value for --rev}"; shift 2 ;;
        --force)   force_install=1; shift ;;
        --cargo)   cargo_only=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *)         echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

selected_refs=0
[[ -n "$git_branch" ]] && selected_refs=$((selected_refs + 1))
[[ -n "$git_tag" ]] && selected_refs=$((selected_refs + 1))
[[ -n "$git_rev" ]] && selected_refs=$((selected_refs + 1))

if [[ $selected_refs -gt 1 ]]; then
    echo "Use only one of --branch, --tag, or --rev" >&2
    exit 1
fi

if [[ "$cargo_only" == "0" && -z "$git_branch" && -z "$git_rev" ]]; then
    install_dir="${install_root:+$install_root/bin}"
    install_dir="${install_dir:-/usr/local/bin}"

    if target="$(detect_target)"; then
        if try_binary_install "$git_tag" "$target" "$install_dir"; then
            exit 0
        fi
        echo "Binary download failed, falling back to cargo install..."
    fi
fi

cargo_install
