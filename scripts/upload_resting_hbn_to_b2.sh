#!/usr/bin/env bash

set -Eeuo pipefail

# Download only the recordings named by the canonical Age manifest, preserving
# the R1..R10/download/... layout expected by the benchmark code, then copy the
# materialized subset to Backblaze B2 using rclone's native B2 backend.

WORK_ROOT="${WORK_ROOT:-/workspace/hbn_subset}"
MANIFEST="${MANIFEST:-${WORK_ROOT}/age_medium_500_resting_manifest.csv}"
B2_BUCKET="${B2_BUCKET:-eeg-benchmark}"
B2_PREFIX="${B2_PREFIX:-dataset}"
HBN_S3_ROOT="${HBN_S3_ROOT:-https://s3.amazonaws.com/openneuro.org}"
LOG_FILE="${LOG_FILE:-/workspace/b2_resting_upload.log}"

DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-5}"
DOWNLOAD_RETRY_MAX_TIME="${DOWNLOAD_RETRY_MAX_TIME:-600}"
TRANSFERS="${TRANSFERS:-4}"
CHECKERS="${CHECKERS:-8}"

log() {
    printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$LOG_FILE"
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || {
        log "ERROR: required command not found: $1"
        exit 127
    }
}

download_file() {
    local url="$1"
    local output="$2"
    local temporary="${output}.part"

    if [[ -s "$output" ]]; then
        log "SKIP existing $output"
        return
    fi

    mkdir -p "$(dirname "$output")"
    log "GET $url"
    curl --fail --location --silent --show-error \
        --retry "$DOWNLOAD_RETRIES" \
        --retry-all-errors \
        --retry-delay 5 \
        --retry-max-time "$DOWNLOAD_RETRY_MAX_TIME" \
        --connect-timeout 30 \
        --output "$temporary" \
        "$url"
    mv "$temporary" "$output"
}

[[ -n "${B2_ACCOUNT:-}" ]] || { log 'ERROR: B2_ACCOUNT is not set'; exit 2; }
[[ -n "${B2_KEY:-}" ]] || { log 'ERROR: B2_KEY is not set'; exit 2; }
[[ -f "$MANIFEST" ]] || { log "ERROR: manifest not found: $MANIFEST"; exit 2; }

require_command curl
require_command rclone
require_command sha256sum

mkdir -p "$WORK_ROOT"

declare -A OPENNEURO_DATASETS=(
    [1]=ds005505
    [2]=ds005506
    [3]=ds005507
    [4]=ds005508
    [5]=ds005509
    [6]=ds005510
    [7]=ds005511
    [8]=ds005512
    [9]=ds005514
    [10]=ds005515
)

manifest_sha256="$(sha256sum "$MANIFEST" | awk '{print $1}')"
printf '%s  %s\n' "$manifest_sha256" "$(basename "$MANIFEST")" \
    > "${MANIFEST}.sha256"

log "manifest=$MANIFEST sha256=$manifest_sha256"
log "bucket=$B2_BUCKET prefix=$B2_PREFIX"

# Keep release-level participant metadata. It is small but required by the
# official study layout and useful for independent validation.
for release_number in $(seq 1 10); do
    release="R${release_number}"
    release_root="${HBN_S3_ROOT}/${OPENNEURO_DATASETS[$release_number]}"
    download_file \
        "${release_root}/participants.tsv" \
        "${WORK_ROOT}/${release}/download/participants.tsv"
done

# The canonical manifest has one row per selected subject. Only the resting
# recording and its four BIDS/EEGLAB sidecars are copied; other task files are
# intentionally excluded.
while IFS=, read -r release subject _rest; do
    [[ "$release" == "release" ]] && continue
    [[ -n "$release" && -n "$subject" ]] || continue

    release_number="${release#R}"
    source_root="${HBN_S3_ROOT}/${OPENNEURO_DATASETS[$release_number]}"
    target_dir="${WORK_ROOT}/${release}/download/${subject}/eeg"
    base="${subject}_task-RestingState"

    download_file \
        "${source_root}/${subject}/eeg/${base}_eeg.set" \
        "${target_dir}/${base}_eeg.set"
    download_file \
        "${source_root}/${subject}/eeg/${base}_eeg.json" \
        "${target_dir}/${base}_eeg.json"
    download_file \
        "${source_root}/${subject}/eeg/${base}_channels.tsv" \
        "${target_dir}/${base}_channels.tsv"
    download_file \
        "${source_root}/${subject}/eeg/${base}_events.tsv" \
        "${target_dir}/${base}_events.tsv"
done < <(awk -F, 'NR > 1 { print $1 "," $2 }' "$MANIFEST")

log 'uploading materialized resting subset to B2'
for release_number in $(seq 1 10); do
    release="R${release_number}"
    rclone copy "${WORK_ROOT}/${release}" ":b2:${B2_BUCKET}/${B2_PREFIX}/${release}" \
        --b2-account "$B2_ACCOUNT" \
        --b2-key "$B2_KEY" \
        --transfers "$TRANSFERS" \
        --checkers "$CHECKERS" \
        --stats 30s \
        --log-level INFO \
        --log-file "$LOG_FILE"
done

log 'uploading manifest and checksum'
rclone copyto "$MANIFEST" ":b2:${B2_BUCKET}/manifests/$(basename "$MANIFEST")" \
    --b2-account "$B2_ACCOUNT" \
    --b2-key "$B2_KEY"
rclone copyto "${MANIFEST}.sha256" ":b2:${B2_BUCKET}/manifests/$(basename "${MANIFEST}.sha256")" \
    --b2-account "$B2_ACCOUNT" \
    --b2-key "$B2_KEY"

log 'upload completed'
rclone size ":b2:${B2_BUCKET}/${B2_PREFIX}" \
    --b2-account "$B2_ACCOUNT" \
    --b2-key "$B2_KEY"
