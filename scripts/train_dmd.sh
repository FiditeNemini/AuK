#!/usr/bin/env bash
# DMD distillation launcher. Mirrors scripts/train.sh conventions.
set -e

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# paths & resources
train_jsonl=agent_space/genshin_zh/train.jsonl
config=ckpts/AuK/config.yaml
teacher_ckpt=ckpts/AuK/auk_base.safetensors
output_dir=ckpts/auk_dmd
initialized_student_path=""
num_gpus=8
attn_backend=torch

# DMDConfig
max_student_updates=2500
frames_threshold=10000
max_samples=24
dataloader_num_workers=2
logging_steps=5
save_per_student_updates=500
last_per_student_updates=100
teacher_cfg_scale=4.0
regression_weight=1.0

mkdir -p "$output_dir" logs/dmd
tag="$(date +'%Y%m%d_%H%M%S')"
log="$REPO/logs/dmd/${tag}_dmd.log"
run_sh="$(cd "$output_dir" && pwd)/${tag}_run.sh"

if [ -n "$initialized_student_path" ]; then
  initialized_student_arg="--initialized_student_path ${initialized_student_path}"
else
  initialized_student_arg=""
fi

cat <<EOF > "$run_sh"
#!/usr/bin/env bash
set -e
cd $REPO
export VIRTUAL_ENV=$REPO/.venv
export PATH=$REPO/.venv/bin:\$PATH
export PYTHONPATH=$REPO/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="ignore"
export NUMBA_CACHE_DIR=/tmp/numba_cache_auk_dmd

$REPO/.venv/bin/torchrun --nproc_per_node ${num_gpus} --nnodes 1 \\
  -m auk.train.distill.dmd.train \\
  --train_jsonl ${train_jsonl} \\
  --config ${config} \\
  --teacher_ckpt ${teacher_ckpt} \\
  --output_dir ${output_dir} \\
  ${initialized_student_arg} \\
  --attn_backend ${attn_backend} \\
  --max_student_updates ${max_student_updates} \\
  --frames_threshold ${frames_threshold} \\
  --max_samples ${max_samples} \\
  --dataloader_num_workers ${dataloader_num_workers} \\
  --logging_steps ${logging_steps} \\
  --save_per_student_updates ${save_per_student_updates} \\
  --last_per_student_updates ${last_per_student_updates} \\
  --teacher_cfg_scale ${teacher_cfg_scale} \\
  --regression_weight ${regression_weight}
EOF

chmod +x "$run_sh"
echo "run script: $run_sh"
echo "log:        $log"
nohup bash "$run_sh" > "$log" 2>&1 &
echo "pid:        $!"
