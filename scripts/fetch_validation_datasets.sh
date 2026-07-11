#!/bin/sh

set -eu

repo_root=$(cd -- "$(dirname -- "$0")/.." && pwd)
destination=${1:-"$repo_root/data/uci"}
tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/muxer-datasets.XXXXXX")

cleanup() {
    rm -rf -- "$tmp_dir"
}
trap cleanup EXIT HUP INT TERM

if ! command -v curl >/dev/null 2>&1; then
    printf '%s\n' 'error: curl is required' >&2
    exit 1
fi
if ! command -v unzip >/dev/null 2>&1; then
    printf '%s\n' 'error: unzip is required' >&2
    exit 1
fi

fetch_archive() {
    name=$1
    url=$2
    archive="$tmp_dir/$name.zip"
    target="$destination/$name"

    mkdir -p -- "$target"
    printf 'downloading %-10s %s\n' "$name" "$url"
    curl --fail --location --silent --show-error "$url" --output "$archive"
    unzip -oq "$archive" -d "$target"
    for nested in "$target"/*.zip; do
        if [ -f "$nested" ]; then
            unzip -oq "$nested" -d "$target"
        fi
    done
}

# All sources are UCI Machine Learning Repository archives. They remain outside
# Git; this script only creates local files under the ignored data/ directory.
fetch_archive mushroom 'https://archive.ics.uci.edu/static/public/73/mushroom.zip'
fetch_archive car 'https://archive.ics.uci.edu/static/public/19/car+evaluation.zip'
fetch_archive bank 'https://archive.ics.uci.edu/static/public/222/bank%2Bmarketing.zip'
fetch_archive wine 'https://archive.ics.uci.edu/static/public/186/wine%2Bquality.zip'
fetch_archive adult 'https://archive.ics.uci.edu/static/public/2/adult.zip'

printf 'datasets ready under %s\n' "$destination"
