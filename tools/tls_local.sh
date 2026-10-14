#!/usr/bin/env bash
# A tiny certificate authority, for reaching nnstudio from your own devices.
#
#   bash tools/tls_local.sh             # make .certs/ (safe to re-run)
#   bash tools/tls_local.sh 10.8.0.9    # …adding that address or name to the cert
#   bash tools/tls_local.sh --force     # re-make the server certificate only,
#                                       #   keeping the CA the devices trust
#   bash tools/tls_local.sh --new-ca    # a brand-new authority (every device
#                                       #   must trust the new ca.crt again)
#
# Writes .certs/ca.crt (the authority — trust this one file on every device you
# browse from) and .certs/server.crt + .certs/server.key (what nn_server
# serves). The directory is git-ignored and the keys are 0600. The certificate
# answers to localhost, this host's name, every address the machine holds, and
# whatever you name on the command line.
#
# Afterwards: restart the service, and open https://<address>:8124 — a secure
# origin, which is where the browser lets the sealing run.
set -e
cd "$(dirname "$0")/.."
DIR=.certs
FORCE=""
NEW_CA=""
while [ -n "${1:-}" ]; do
  case "$1" in
    --force) FORCE=1; shift ;;
    --new-ca) NEW_CA=1; shift ;;
    *) break ;;
  esac
done
if [ -e "$DIR/server.crt" ] && [ -z "$FORCE" ]; then
  echo "$DIR already holds a certificate — pass --force to replace it"
  exit 0
fi
command -v openssl >/dev/null || { echo "openssl is needed (apt install openssl)"; exit 1; }
mkdir -p "$DIR"
{
  echo "DNS:localhost"
  echo "DNS:$(hostname)"
  echo "IP:127.0.0.1"
  echo "IP:::1"
  for a in $(hostname -I 2>/dev/null) "$@"; do
    case "$a" in
      *[0-9].[0-9]*|*:*) echo "IP:$a" ;;
      *) echo "DNS:$a" ;;
    esac
  done
} | sort -u > "$DIR/san.txt"
ALTS=$(paste -sd, "$DIR/san.txt")
if [ ! -e "$DIR/ca.crt" ] || [ -n "$NEW_CA" ]; then
  openssl req -x509 -newkey rsa:3072 -sha256 -days 3650 -nodes \
    -keyout "$DIR/ca.key" -out "$DIR/ca.crt" -subj "/CN=nnstudio local CA" \
    -addext "basicConstraints=critical,CA:TRUE" \
    -addext "keyUsage=critical,keyCertSign,cRLSign" 2>/dev/null
fi
openssl req -newkey rsa:2048 -sha256 -nodes \
  -keyout "$DIR/server.key" -out "$DIR/server.csr" -subj "/CN=nnstudio" 2>/dev/null
printf 'subjectAltName=%s\n' "$ALTS" > "$DIR/server.ext"
openssl x509 -req -in "$DIR/server.csr" -CA "$DIR/ca.crt" -CAkey "$DIR/ca.key" \
  -CAcreateserial -days 825 -sha256 -extfile "$DIR/server.ext" \
  -out "$DIR/server.crt" 2>/dev/null
rm -f "$DIR/server.csr" "$DIR/server.ext" "$DIR/ca.srl"
chmod 600 "$DIR"/*.key
echo "made $DIR/server.crt, answering to:"
sed 's/^/  /' "$DIR/san.txt"
echo "trust $DIR/ca.crt on each device, restart nnstudio.service, then open https://<address>:8124"
