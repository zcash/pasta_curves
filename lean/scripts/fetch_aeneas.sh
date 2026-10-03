#!/usr/bin/env bash
# Fetch the Charon and Aeneas binaries that `gen_portable.sh` runs. They come from the release that
# Aeneas' CI built at the revision that `lean/lakefile.toml` pins. The script downloads the release
# for this platform, checks it against the SHA-256 recorded below, and unpacks the binaries into DIR
# (default `lean/work/aeneas`). It skips the download when DIR already holds this release. Charon
# runs the Rust compiler of the nightly toolchain that the release names, so the script also
# installs that toolchain with rustup, in its minimal profile. Run it again whenever the pin
# moves: `gen_portable.sh` fails while the default DIR holds another release.
#
# Usage, from anywhere in the checkout: lean/scripts/fetch_aeneas.sh [DIR | --tag]
# It prints the settings for `gen_portable.sh`, which needs them only for a DIR other than the
# default: CHARON=DIR/charon AENEAS=DIR/aeneas. With `--tag`, it prints the release's tag instead,
# and fetches nothing.
set -euo pipefail
cd "$(dirname "$0")/../.."

# The release built at the pinned revision, and the SHA-256 of its tarball for each platform that it
# supports. Moving the pin means moving these to the release built at the new revision.
tag=nightly-2026.09.26-b86120d
if [ "${1:-}" = "--tag" ]; then
  echo "$tag"
  exit 0
fi
dir=${1:-lean/work/aeneas}
case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)
    platform=linux-x86_64
    sha256=583c14726c9e4e9aebd624eb84e750b51167ec39dd9db49bd4f655546956a391 ;;
  Linux-aarch64)
    platform=linux-aarch64
    sha256=11bc02245bcdd7764e5f3f0427323bf1751b8e1a7096ed1182755451cf7645e7 ;;
  Darwin-arm64)
    platform=macos-aarch64
    sha256=8739a2425d639badd629a58905027f9009cbdae77c78f7a7f6496dee021be350 ;;
  *)
    echo "error: Aeneas does not publish a release for $(uname -s) on $(uname -m)" >&2
    exit 1 ;;
esac

# A release's tag ends with the abbreviated revision that it was built at.
rev=$(awk '/^\[\[require\]\]/ { aeneas = 0 } /^name = "aeneas"/ { aeneas = 1 }
  aeneas && /^rev = / { gsub(/"/, "", $3); print $3 }' lean/lakefile.toml)
case "$rev" in
  "${tag##*-}"?*) ;;
  *)
    echo "error: release $tag was not built at the pinned revision ${rev:-(none found)}" >&2
    exit 1 ;;
esac

mkdir -p "$dir"
dir=$(cd "$dir" && pwd)
if [ "$(cat "$dir/release" 2>/dev/null)" != "$tag" ]; then
  download=$(mktemp -d)
  trap 'rm -rf "$download"' EXIT
  asset=aeneas-$platform.tar.gz
  curl -fsSL -o "$download/$asset" \
    "https://github.com/AeneasVerif/aeneas/releases/download/$tag/$asset"
  echo "$sha256  $asset" > "$download/$asset.sha256"
  if command -v sha256sum > /dev/null; then
    (cd "$download" && sha256sum -c "$asset.sha256") >&2
  else
    (cd "$download" && shasum -a 256 -c "$asset.sha256") >&2
  fi
  # The release includes Aeneas' Lean library, which the Lean package fetches for itself.
  tar -xzf "$download/$asset" -C "$dir" --exclude backends
  echo "$tag" > "$dir/release"
fi

channel=$(awk -F'"' '/^channel = / { print $2 }' "$dir/rust-toolchain")
rustup toolchain install "$channel" --profile minimal >&2
echo "CHARON=$dir/charon AENEAS=$dir/aeneas"
