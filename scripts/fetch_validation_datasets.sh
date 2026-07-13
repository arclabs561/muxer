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

sha256_file() (
    if command -v sha256sum >/dev/null 2>&1; then
        digest=$(sha256sum "$1")
    elif command -v shasum >/dev/null 2>&1; then
        digest=$(shasum -a 256 "$1")
    else
        printf '%s\n' 'error: sha256sum or shasum is required' >&2
        exit 1
    fi
    printf '%s\n' "${digest%% *}"
)

verify_archive() (
    archive=$1
    expected_bytes=$2
    expected_sha256=$3
    actual_bytes=$(wc -c < "$archive")
    actual_sha256=$(sha256_file "$archive")

    if [ "$actual_bytes" -ne "$expected_bytes" ]; then
        printf 'error: %s has %s bytes, expected %s\n' \
            "$archive" "$actual_bytes" "$expected_bytes" >&2
        exit 1
    fi
    if [ "$actual_sha256" != "$expected_sha256" ]; then
        printf 'error: %s has SHA-256 %s, expected %s\n' \
            "$archive" "$actual_sha256" "$expected_sha256" >&2
        exit 1
    fi
)

fetch_archive() (
    name=$1
    url=$2
    expected_bytes=$3
    expected_sha256=$4
    archive="$destination/raw/$name.zip"
    target="$destination/$name"
    marker="$target/.muxer-source-sha256"

    mkdir -p -- "$destination/raw"
    if [ -f "$archive" ]; then
        verify_archive "$archive" "$expected_bytes" "$expected_sha256"
        printf 'verified    %-10s %s\n' "$name" "$archive"
    elif [ -e "$archive" ]; then
        printf 'error: archive path is not a file: %s\n' "$archive" >&2
        exit 1
    else
        download="$tmp_dir/$name.zip"
        printf 'downloading %-10s %s\n' "$name" "$url"
        curl --fail --location --silent --show-error "$url" --output "$download"
        verify_archive "$download" "$expected_bytes" "$expected_sha256"
        mv -- "$download" "$archive"
    fi

    if [ -f "$marker" ]; then
        IFS= read -r marker_sha256 < "$marker"
        if [ "$marker_sha256" != "$expected_sha256" ]; then
            printf 'error: %s was extracted from a different archive; move it aside and retry\n' \
                "$target" >&2
            exit 1
        fi
        printf 'verified    %-10s %s\n' "$name" "$target"
        return
    fi
    if [ -e "$target" ]; then
        printf 'error: refusing to overwrite unverified dataset directory: %s\n' \
            "$target" >&2
        exit 1
    fi

    extracted="$tmp_dir/extracted-$name"
    mkdir -p -- "$extracted"
    unzip -q "$archive" -d "$extracted"
    for nested in "$extracted"/*.zip; do
        if [ -f "$nested" ]; then
            unzip -q "$nested" -d "$extracted"
        fi
    done
    printf '%s\n' "$expected_sha256" > "$extracted/.muxer-source-sha256"
    mv -- "$extracted" "$target"
)

# These byte counts and digests pin the UCI archives used by the trace builder.
# Raw archives and extracted files remain under the ignored data/ directory.
fetch_archive mushroom 'https://archive.ics.uci.edu/static/public/73/mushroom.zip' \
    141318 face32f32647e0d939f6233f36dd30dd5d619ae9f3f9b8e10bea4ac7e1f60b1a
fetch_archive car 'https://archive.ics.uci.edu/static/public/19/car+evaluation.zip' \
    6342 1559d51dcf327f4f8c71b711ceed7fd95a382fc8d8e1998667f4f23b82860403
fetch_archive bank 'https://archive.ics.uci.edu/static/public/222/bank%2Bmarketing.zip' \
    1023843 e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4
fetch_archive wine 'https://archive.ics.uci.edu/static/public/186/wine%2Bquality.zip' \
    91353 3ed56667f4b828242bd732d7d1dd7f2861e54432239d7fa63877014cbb0304d4
fetch_archive adult 'https://archive.ics.uci.edu/static/public/2/adult.zip' \
    620237 7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb

printf 'datasets ready under %s\n' "$destination"
