#!/usr/bin/env bash
# Fetch the Charon and Aeneas binaries that `gen_aeneas.sh` runs. They come from the release that
# Aeneas' CI built at the revision that `lean/lakefile.toml` pins. The script downloads the release's
# tarball for this platform into DIR (default `lean/work/aeneas`), checks it against the SHA-256
# recorded below, and unpacks the binaries there. It keeps the tarball, and downloads it again only
# when DIR lacks it or it fails the check, but it checks and unpacks it on every run, so that the
# binaries in DIR are always those of the checked release. Charon runs the Rust compiler of the
# nightly toolchain that the release names, so the script also installs that toolchain with rustup,
# in its minimal profile. Run it again whenever the pin moves: `gen_aeneas.sh` fails while the
# default DIR holds another release.
#
# Usage, from anywhere in the checkout: lean/scripts/fetch_aeneas.sh [DIR | --tag]
# A relative DIR is taken from the current directory. The script prints the settings for
# `gen_aeneas.sh`, which needs them only for a DIR other than the default: CHARON=DIR/charon
# AENEAS=DIR/aeneas. With `--tag`, it prints the release's tag instead, once it has checked that the
# release was built at the pinned revision, and fetches nothing.
set -euo pipefail

# DIR against the caller's directory, before the script moves to the checkout's root.
case "${1:-}" in
  "" | --tag) dir= ;;
  /*) dir=$1 ;;
  *) dir=$PWD/$1 ;;
esac
cd "$(dirname "$0")/../.."
dir=${dir:-lean/work/aeneas}

# The release built at the pinned revision, and the SHA-256 of its tarball for each platform that it
# supports. Moving the pin means moving these to the release built at the new revision.
tag=nightly-2026.09.26-b86120d

# A release's tag ends with the abbreviated revision that it was built at.
rev=$(awk '/^\[\[require\]\]/ { aeneas = 0 } /^name = "aeneas"/ { aeneas = 1 }
  aeneas && /^rev = / { gsub(/"/, "", $3); print $3 }' lean/lakefile.toml)
case "$rev" in
  "${tag##*-}"?*) ;;
  *)
    echo "error: release $tag was not built at the pinned revision ${rev:-(none found)}" >&2
    exit 1 ;;
esac
if [ "${1:-}" = "--tag" ]; then
  echo "$tag"
  exit 0
fi

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

mkdir -p "$dir"
dir=$(cd "$dir" && pwd)
asset=aeneas-$platform.tar.gz
echo "$sha256  $asset" > "$dir/$asset.sha256"

# checked: whether DIR's tarball has the recorded SHA-256.
checked() {
  if command -v sha256sum > /dev/null; then
    (cd "$dir" && sha256sum -c "$asset.sha256") >&2
  else
    (cd "$dir" && shasum -a 256 -c "$asset.sha256") >&2
  fi
}

if [ ! -f "$dir/$asset" ] || ! checked 2> /dev/null; then
  curl -fsSL -o "$dir/$asset" \
    "https://github.com/AeneasVerif/aeneas/releases/download/$tag/$asset"
fi
checked
# The release includes Aeneas' Lean library, which the Lean package fetches for itself.
tar -xzf "$dir/$asset" -C "$dir" --exclude backends
echo "$tag" > "$dir/release"

channel=$(awk -F'"' '/^channel = / { print $2 }' "$dir/rust-toolchain")
rustup toolchain install "$channel" --profile minimal >&2
echo "CHARON=$dir/charon AENEAS=$dir/aeneas"
