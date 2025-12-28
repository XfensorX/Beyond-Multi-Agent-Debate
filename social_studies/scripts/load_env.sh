#!/usr/bin/env bash
set -euo pipefail

# Loads an env file based on the current hostname.
# Rules:
# - Base env file name is: .env.<hostname>
# - If hostname contains "-node<digits>" suffix, strip it (e.g. pascal-node12 -> pascal)
# - If a matching env file doesn't exist, exit with an error.


if [[ ! -f "scripts/environment/.env.example" ]]; then
  echo "ERROR: The load_env.sh script has to be executed from the project directory." >&2
  exit 1
fi

raw_host="$(hostname -s 2>/dev/null || hostname)"
raw_host="${raw_host//$'\n'/}"

# Strip optional "-node<digits>" suffix
base_host="$(printf '%s' "$raw_host" | sed -E 's/-node[0-9]+$//')"

env_file="scripts/environment/.env.${base_host}"

if [[ ! -f "$env_file" ]]; then
  echo "Searched for $env_file in pwd=$(pwd)" >&2
  echo "ERROR: env file not found: $env_file (hostname was: $raw_host, normalized: $base_host)" >&2
  exit 1
fi
set -a
source "$env_file"
set +a