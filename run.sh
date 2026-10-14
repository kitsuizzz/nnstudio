#!/usr/bin/env bash
# Run nnstudio on http://localhost:8124 (pass --port to change it).
# A .certs/ pair from tools/tls_local.sh is served too, on the same port.
set -e
cd "$(dirname "$0")"
if [ -e .certs/server.crt ] && [ -e .certs/server.key ]; then
  exec .venv/bin/python nn_server.py \
    --tls-cert .certs/server.crt --tls-key .certs/server.key "$@"
fi
exec .venv/bin/python nn_server.py "$@"
