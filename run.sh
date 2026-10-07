#!/usr/bin/env bash
# Run nnstudio on http://localhost:8124 (pass --port to change it).
set -e
cd "$(dirname "$0")"
exec .venv/bin/python nn_server.py "$@"
