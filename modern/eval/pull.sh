#!/bin/bash
# Fetch a run directory (small files) from a board into results/.
# usage: GW=... ./pull.sh <port> <remote run dir> <local dir>
set -e
port=$1; remote=$2; local=$3
mkdir -p "$local"
/tmp/jraw.sh "$port" "cd '$remote' && tar cf - --exclude 'loaded_*' --exclude 'ready_*' --exclude go_warmup --exclude sampler.stop --exclude 'phase.json' ." | tar xf - -C "$local"
ls "$local"
