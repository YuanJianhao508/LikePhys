#!/usr/bin/env bash
GPUS=(0 1 2 3 4 5 6 7)
NGPUS=${#GPUS[@]}

SEEDS=(42)
DATASETS=(ball_drop ball_collision pendulum block_slide pyramid fluid faucet river flag cloth shadow shadowm)
MODELS=(animatediff zeroscope modelscope wan2.1-T2V-1.3b hunyuan_t2v ltx-0.9.5 animatediff_sdxl cogvideox mochi cogvideox-5b wan2.1-T2V-14b)
FLAGS=("--guidance_scale")

declare -A GPU_PIDS

get_free_gpu() {
  for idx in "${!GPUS[@]}"; do
    gpu=${GPUS[$idx]}
    pid=${GPU_PIDS[$gpu]}
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
      echo "$gpu"
      return
    fi
  done
  echo ""
}

for seed in "${SEEDS[@]}"; do
  for model in "${MODELS[@]}"; do
    for data in "${DATASETS[@]}"; do
    # for num_frames in "${NUM_FRAMES[@]}"; do
      for flag in "${FLAGS[@]}"; do

        # Wait for a free GPU
        while true; do
          gpu=$(get_free_gpu)
          if [ -n "$gpu" ]; then
            break
          fi
          sleep 1
        done

        echo "→ GPU $gpu ← model=$model, data=$data, seed=$seed, flag=$flag"
        CUDA_VISIBLE_DEVICES=$gpu \
          python evaluator.py --model="$model" --data="$data" --seed="$seed" $flag --tag_name="final" &
        GPU_PIDS[$gpu]=$!
      done
    done
  done
# done
done

wait
echo "✅ All evaluations done."
