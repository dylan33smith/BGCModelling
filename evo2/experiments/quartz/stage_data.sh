#!/usr/bin/env bash
# Stage weights + build the long-context dataset on the Quartz LOGIN node.
# splits_core must already be rsync'd from the lab box (see message below).
set -uo pipefail
export HF_HOME=${HF_HOME:-/N/slate/$USER/hf_cache}
module load conda; conda activate bgcmodel
DATA=/N/slate/$USER/bgcmodel_data
mkdir -p "$DATA"

echo "==> Evo2 7B weights -> $HF_HOME"
huggingface-cli download arcinstitute/evo2_7b_262k

if [ ! -f "$DATA/splits_core/train.jsonl" ]; then
  cat <<EOF
!! splits_core not found at $DATA/splits_core/
   Push it from the LAB box (run THIS on the lab box, not Quartz):
     rsync -avP /data2/ds85/bgcmodel_data/splits_core/ \\
       $USER@quartz.uits.iu.edu:$DATA/splits_core/
   Then re-run this script.
EOF
  exit 1
fi

echo "==> build mega-only dataset (ALL lengths — every core kept whole at long L)"
python - "$DATA" <<'PY'
import json,sys
DATA=sys.argv[1]; mega={'NRPS','PKS','PKS_NRPS_HYBRID'}
o=open(f'{DATA}/mega_all.jsonl','w'); n=0
for l in open(f'{DATA}/splits_core/train.jsonl'):
    if json.loads(l).get('compound_class') in mega: o.write(l); n+=1
print('mega cores written:',n,'(~209 Mbp, max len 262144)')
PY
python evo2/scripts/build_chunk_index.py --jsonl "$DATA/mega_all.jsonl" \
  --max-seq-len 262144 --chunk-overlap 2048
echo "==> done. Dataset: $DATA/mega_all.jsonl (+ .lengths sidecar)"
