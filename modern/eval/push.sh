#!/bin/bash
# Upload the harness (eval/neuos minus results) to <workdir>/harness on a board.
# usage: GW=... ./push.sh <port>       (uses /tmp/jssh2.sh)
set -e
port=$1
cd "$(dirname "$0")"
tar cf - --no-mac-metadata --exclude results --exclude __pycache__ --exclude '._*' . \
  | /tmp/jssh2.sh "$port" 'mkdir -p /experiment/zexin/neuos_eval/harness && cd /experiment/zexin/neuos_eval/harness && tar xmf - && echo pushed'
