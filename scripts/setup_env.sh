#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DIFFUSERS_COMMIT="10dc589a942a38a53d598275b0b421b69a6e03cc"
VENV_DIR="${VENV_DIR:-.venv}"

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements/lock.txt

if [[ ! -d ".third_party/diffusers/.git" ]]; then
  mkdir -p .third_party
  git clone https://github.com/huggingface/diffusers.git .third_party/diffusers
fi

git -C .third_party/diffusers fetch --all --tags
git -C .third_party/diffusers checkout "$DIFFUSERS_COMMIT"

# Install exact local sources without dependency re-resolution.
python -m pip install --no-deps -e .third_party/diffusers
python -m pip install --no-deps -e .

echo "Environment ready."
echo "Activate with: source $VENV_DIR/bin/activate"
