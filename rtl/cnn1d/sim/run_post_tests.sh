#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
out=$(mktemp -d "${TMPDIR:-/tmp}/ecg-post-tests.XXXXXX")
trap 'rm -rf "$out"' EXIT
compiler=${IVERILOG:-iverilog}
runtime=${VVP:-vvp}
flags=(-g2012)
if [[ -n ${IVL_BASE:-} ]]; then flags+=(-B "$IVL_BASE"); fi
python3 - "$out/vectors.txt" <<'PY'
import random, sys
rng = random.Random(16)
cases = [(a,b,s,r) for a in [-(2**31),-1000,-5,-3,-1,0,1,3,5,126,127,128,1000,2**31-1]
         for b in [-(2**31),0,2**31-1] for s in [-128,-8,-1,0,1,3,31,32,33,34,127] for r in [0,1]]
cases += [(rng.randrange(-2**31,2**31),rng.randrange(-2**31,2**31),rng.randrange(-128,128),rng.randrange(2)) for _ in range(1000)]
with open(sys.argv[1], 'w') as f:
    for a,b,s,r in cases:
        v=a+b
        v=(v+(1<<(s-1)))>>s if s>0 else v<<(-s)
        v=max(0 if r else -127,min(127,v))
        f.write(f'{a} {b} {s} {r} {v}\n')
print(f'Generated {len(cases)} integer golden cases')
PY
"$compiler" "${flags[@]}" -s tb_post_process -o "$out/post.vvp" -f filelist.f tb/tb_post_process.v
"$runtime" "$out/post.vvp" "+vectors=$out/vectors.txt"
"$compiler" "${flags[@]}" -s tb_conv1d_tile_top -o "$out/tile.vvp" -f filelist.f tb/tb_conv1d_tile_top.v
"$runtime" "$out/tile.vvp" > "$out/tile.log"
if grep -Eq '\[FAIL\]|FAILED|TIMEOUT' "$out/tile.log"; then cat "$out/tile.log"; exit 1; fi
grep 'ALL TESTS PASSED' "$out/tile.log"
