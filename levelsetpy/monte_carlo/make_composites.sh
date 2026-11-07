#!/usr/bin/env bash
# Regenerate the three matched 2x3 composites (rockets, dubins_relative, murmuration).
#
#   ./make_composites.sh --pdf ~/Documents/Papers/MSRYears/TOMS/levtoms.pdf \
#                        --anim-dir ~/Documents/blog/downloads/Papers/assets/anim/originals_1x3 \
#                        --work /path/to/scratch
#
# Outputs: WORK/fields/*.npz (value fields), WORK/composites/*.gif (+ _poster.png).
# Solves are slow (tens of minutes each on a laptop). Re-running skips nothing.
set -euo pipefail

PDF=""; ANIM=""; WORK=""; N=160; STEPS=400
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pdf) PDF="$2"; shift 2 ;;
    --anim-dir) ANIM="$2"; shift 2 ;;
    --work) WORK="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$PDF" && -n "$ANIM" && -n "$WORK" ]] || { sed -n '2,8p' "$0"; exit 2; }

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"
export PYTHONPATH="$HERE"
cd "$HERE"

echo "[1/3] Fig. 4 decals"
"$PY" -m hjgauss.render.decals --pdf "$PDF"

echo "[2/3] value fields (benchmark = reference, HJ-Gauss = full)"
mkdir -p "$WORK/fields" "$WORK/composites"
for case in rockets dubins_relative murmuration; do
  for kind in reference full; do
    "$PY" -m hjgauss.validation.cases --suite games --cases "$case" \
      --n "$N" --steps "$STEPS" --kind "$kind" --out "$WORK/fields"
  done
done

echo "[3/3] composites"
for case in rockets dubins_relative murmuration; do
  "$PY" -m hjgauss.render.matched3col --case "$case" \
    --hj-root "$WORK/fields" --anim-dir "$ANIM" \
    --out "$WORK/composites/roysoc_${case/dubins_relative/dubins}.gif"
done
echo "done: $WORK/composites"
