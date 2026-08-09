#!/usr/bin/env bash
# mp4 -> GIF for the training demo artifacts.
#
# play_fixed_vel.py --video writes mp4 via gym RecordVideo; this turns one into
# the GIF that goes on the PR. Two-pass palettegen/paletteuse, because a single
# global palette on a mostly-grey Isaac render bands badly on the shadows.
#
# In the repo rather than a scratch dir on purpose: the previous version of this
# lived in /tmp and did not survive a reboot, which cost a rebuild.
#
# Usage: make_gif.sh <input.mp4> <output.gif> [fps] [width] [speed]
#   fps    default 20
#   width  default 640 (height auto, preserves aspect)
#   speed  default 1.0; 2.0 plays twice as fast
set -euo pipefail

IN="${1:?usage: make_gif.sh <input.mp4> <output.gif> [fps] [width] [speed]}"
OUT="${2:?output .gif path required}"
FPS="${3:-20}"
WIDTH="${4:-640}"
SPEED="${5:-1.0}"

[ -f "$IN" ] || { echo "no such input: $IN" >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "ffmpeg not found" >&2; exit 1; }

PAL="$(mktemp -t bbpalette.XXXXXX.png)"
trap 'rm -f "$PAL"' EXIT

# setpts before fps so the speed change is resampled, not frame-dropped twice.
FILTERS="setpts=PTS/${SPEED},fps=${FPS},scale=${WIDTH}:-1:flags=lanczos"

ffmpeg -v error -y -i "$IN" -vf "${FILTERS},palettegen=stats_mode=diff" "$PAL"
ffmpeg -v error -y -i "$IN" -i "$PAL" \
  -lavfi "${FILTERS} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" \
  "$OUT"

SIZE=$(du -h "$OUT" | cut -f1)
DUR=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$OUT" 2>/dev/null || echo "?")
echo "wrote $OUT  (${SIZE}, ${DUR}s, ${FPS} fps, ${WIDTH}px, ${SPEED}x)"

# GitHub will not inline a GIF much over ~10 MB; warn rather than fail so the
# caller can decide whether to drop fps or width.
BYTES=$(stat -c %s "$OUT")
if [ "$BYTES" -gt 10485760 ]; then
  echo "WARNING: ${SIZE} exceeds ~10 MB; GitHub may not render it inline." >&2
  echo "         retry with a lower fps or width." >&2
fi
