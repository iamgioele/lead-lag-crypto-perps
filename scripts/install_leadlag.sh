#!/usr/bin/env bash
# Reproducible build of lead-lag 2.3 (MIT) into the project venv.
# Why patched: the PyPI sdist cannot build in pip's isolated environment (no Cython),
# ships a stale generated C file, and its setup.py shells out to `pip install cython`.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
URL="https://files.pythonhosted.org/packages/5d/8b/28153385bc70fa59821ff80457c62de15b438bbdb4c466a972a005fead48/lead-lag-2.3.tar.gz"
SHA="77706b94814c1248e55347050da70e8030e8a111f7b2cdaaa961b9d394c01b89"
curl -sL -o "$WORK/ll.tar.gz" "$URL"
echo "$SHA  $WORK/ll.tar.gz" | shasum -a 256 -c -           # shasum: present on both Linux and macOS
tar -xzf "$WORK/ll.tar.gz" -C "$WORK"
SRC="$WORK/lead-lag-2.3"
rm -f "$SRC/lead_lag/lead_lag_impl.c"                     # force regeneration with Cython 3
sed "s/^os.system('pip install cython')/# removed: build-time pip call/" "$SRC/setup.py" > "$SRC/setup.py.new"
mv "$SRC/setup.py.new" "$SRC/setup.py"                    # (portable: GNU and BSD sed differ on -i)
VIRTUAL_ENV="$ROOT/.venv" uv pip install --no-build-isolation "$SRC"
(cd "$WORK" && "$ROOT/.venv/bin/python" -c "import lead_lag.lead_lag_impl; print('lead_lag compiled OK')")
