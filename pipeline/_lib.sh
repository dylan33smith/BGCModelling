# Shared helpers for the post-training pipeline. Sourced, not executed.
#
# ⚠ SUBSTRATE-DEPENDENT VALUES ARE NEVER LITERALS HERE. They are read from
# bgcbench.model.substrate_config, which carries each value's provenance. Bare literals in
# shell scripts are how `--rank 4` and `--rank 16` both ended up in work/*.sh with nothing
# tying either to a substrate (SPEC 14A).
PY=/home/ds85/.local/share/mamba/envs/bgcmodel/bin/python
REPO=/home/ds85/projects/BCGModelling/.claude/worktrees/bgc-benchmarking-paper-139396
A=/data2/ds85/bgcbench/adapters
D=/data2/ds85/bgcbench/directions
G9=/data2/ds85/bgcbench/g9
CLASSES="TERPENE ARYLPOLYENE RIPP REDOX_COFACTOR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $REPO

# family for a substrate id, and its gated config
fam ()    { case "$1" in evo2*) echo evo2 ;; *) echo genomeocean ;; esac; }
cfgval () { $PY -c "
import sys; sys.path.insert(0,'$REPO')
from bgcbench.model.substrate_config import for_substrate
print(for_substrate('$(fam $1)')['$2'])"; }

# ⚠ WAIT FOR GPU ROOM -- WITH A MEASURED THRESHOLD, NOT AN ESTIMATED ONE.
#   Evo2 generation  29,690 MiB   measured, run.arm, batch 50   -> threshold 28,000
#   GenomeOcean      22,346 MiB   measured, training, this card -> threshold 30,000
#
# ⚠ AN OVER-ESTIMATED THRESHOLD IS A DEADLOCK, NOT A SAFETY MARGIN. GenomeOcean was first
# given 58,000/60,000 MiB from a figure I misread off `nvidia-smi` -- that reading was the
# card's TOTAL usage including other users, not our process. With another user holding 40 GB
# only ~41 GB is ever available, so the guard could never be satisfied: train_go_fx.sh sat
# polling from 16:27 to 05:37 the next morning, 13 hours of idle GPU, and only started
# because that user briefly dropped off. The job itself needs ~22 GB and would have run the
# whole time.
wait_gpu () {
  local need_mib=$1 label=$2
  while true; do
    local free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
    [ "$free" -ge "$need_mib" ] && break
    echo "  waiting for GPU: need ${need_mib} MiB, ${free} free ($label) $(date -Is)" >> $S
    # ⚠ 60s, NOT 300s. With two substrates alternating on one card every handoff costs up to
    # a full poll interval of idle GPU; across ~30 arms that was up to 2.5h of nothing.
    sleep 60
  done
}

step () {  # step NAME CMD...
  local name=$1; shift
  echo "START $name $(date -Is)" >> $S
  "$@" >> $L 2>&1
  local rc=$?
  echo "DONE  $name rc=$rc $(date -Is)" >> $S
  [ $rc -ne 0 ] && echo "  ⚠ $name FAILED — later steps that depend on it will be skipped" >> $S
  return $rc
}

# ⚠ FAIL FAST ON A MISSING ADAPTER. The phase scripts build adapter paths by string
# concatenation (${SUB}_W2_${C}${SUF}), so a naming mismatch produces a path that simply does
# not exist -- and without this it would surface as a load error partway through a multi-hour
# chain, after the earlier arms had already run. Checked once, up front, for the whole phase.
require_adapters () {   # require_adapters <path> [<path> ...]
  local missing=0
  for a in "$@"; do
    if [ ! -e "$a" ]; then echo "  ⛔ MISSING ADAPTER: $a" >> $S; missing=1; fi
  done
  if [ $missing -ne 0 ]; then
    echo "  ⛔ ABORTING PHASE — expected adapters are absent. Check the suffix: Evo2 arms are" >> $S
    echo "     named <arm>_tax_fx (SUF=_tax_fx) and GenomeOcean <arm>_fx (SUF=_fx)." >> $S
    exit 2
  fi
}
