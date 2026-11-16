#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install it, then run this script again." >&2
    exit 1
fi

if (( $# > 1 )); then
    echo "Usage: $0 [--learning]" >&2
    exit 2
fi

case "${1:-}" in
    "")
        set --
        ;;
    --learning)
        set -- --extra learning
        ;;
    -h|--help)
        echo "Usage: $0 [--learning]"
        echo "Add --learning to install the optional PPO dependencies (torch and numpy)."
        exit 0
        ;;
    *)
        echo "Usage: $0 [--learning]" >&2
        exit 2
        ;;
esac

uv sync --python 3.11 "$@"

echo "Environment ready. From this folder, run:"
echo "  source .venv/bin/activate"
echo "  python -m examples.demo --agent rulebased"
