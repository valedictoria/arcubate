#!/bin/bash
# Streams monthly Lichess game files (CC0) through the extractor. The downloads
# are never written to disk. Usage: fetch_games.sh outdir minElo month...
out=$1; elo=$2; shift 2
here=$(dirname "$(readlink -f "$0")")
mkdir -p "$out"
for m in "$@"; do
  [ -s "$out/$m.agg" ] && continue
  echo "== $m $(date +%T)"
  set -o pipefail
  if curl -sS --fail "https://database.lichess.org/standard/lichess_db_standard_rated_$m.pgn.zst" \
     | zstd -dcq | nice -n 19 "$here/book_tool" extract "$out/$m.tmp" "$elo" 36 2>&1 >/dev/null | tail -1
  then mv "$out/$m.tmp" "$out/$m.agg"
  else echo "FAILED $m"; rm -f "$out/$m.tmp"; fi
done
echo "== finished $(date +%T)"
