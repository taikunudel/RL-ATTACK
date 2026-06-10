# A40 (r06g04) — add as a 3rd guard to accelerate the grid

The A40 (46GB, persistent) can host a 3rd guard → a 3rd training stream → faster grid completion.
It's NOT reachable from infodeep, so YOU run the setup on the node; it connects OUT to infodeep
via a reverse tunnel that exposes its guard on `infodeep:8002`, which the orchestrator then uses.

## Step 0 — tell me the disk path (the one missing fact)
Home is only 17GB (< the 24GB model). Run on the A40 node and paste output:
```
df -h /tmp /scratch /data /work "$HOME" 2>/dev/null | grep -v Filesystem
echo "SCRATCH=$SCRATCH TMPDIR=$TMPDIR SLURM_TMPDIR=$SLURM_TMPDIR"
```
Pick a path with >=35GB free; call it $BIG below.

## Step 1 — bootstrap python + deps (no conda/root needed), ON the A40 node
```
BIG=/PICK_A_BIG_DIR            # e.g. /scratch/taikun  (>=35GB free)
mkdir -p "$BIG/hf" "$BIG/mc"
# Miniconda (user-level)
cd "$BIG"; curl -L -o mc.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash mc.sh -b -p "$BIG/mc"; source "$BIG/mc/etc/profile.d/conda.sh"
conda create -y -n guard python=3.11; conda activate guard
pip install "torch>=2.6" transformers flask accelerate
export HF_HOME="$BIG/hf"
# HF auth (gated model). Paste your token:
python -c "from huggingface_hub import login; login('hf_PASTE_YOUR_TOKEN')"
```

## Step 2 — get the server file (scp from infodeep, OR it'll be staged)
```
scp taikun@128.4.10.226:/usa/taikun/rl-attack/rl_atk/attack-genai/serve_guard_cfg.py "$BIG/"
```

## Step 3 — start the guard on the A40 (native bf16; A40 46GB fits easily, MB can be larger)
```
cd "$BIG"
HF_HOME="$BIG/hf" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  GUARD_DTYPE=bfloat16 GUARD_PORT=8000 GUARD_MICRO_BATCH=16 \
  setsid nohup python -u serve_guard_cfg.py > "$BIG/guard.log" 2>&1 < /dev/null &
# wait ~5 min (download+load), then:
curl -s localhost:8000/health    # expect {"dtype":"torch.bfloat16",...,"status":"ok"}
```

## Step 4 — reverse tunnel: expose A40 guard on infodeep:8002
```
ssh -fN -R 8002:localhost:8000 taikun@128.4.10.226
```
Tell me when this is up. Then I:
- verify infodeep `curl localhost:8002/health` + run my byte-exact safe/unsafe match vs the other guards,
- add `"2 http://localhost:8002/v1"` to run_grid_all.sh STREAMS,
- restart the orchestrator → a 3rd stream starts claiming configs.

Net effect: 3 fast streams instead of 2 → ~33% faster to finish all 99.
