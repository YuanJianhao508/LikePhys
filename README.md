# LikePhys: Evaluating intuitive physics understanding in video diffusion models via likelihood preference 

**ICLR 2026**

[[arXiv]](https://arxiv.org/abs/2510.11512) [[Project Page]](https://yuanjianhao508.github.io/LikePhys/) [[Dataset]](https://huggingface.co/datasets/JianhaoDYDY/LikePhys-Benchmark)

## Abstract
*LikePhys*, a training-free method that evaluates intuitive physics in video diffusion models by distinguishing physically valid and impossible videos using the denoising objective as an ELBO-based likelihood surrogate on a curated dataset of valid-invalid pairs.

## Usage

### Quick Start

1. **Setup Environment**:
```bash
# Clone repository
git clone https://github.com/YuanJianhao508/LikePhys.git
cd LikePhys

# Install dependencies
pip install torch torchvision diffusers accelerate transformers
pip install opencv-python pillow numpy matplotlib tqdm

# Download dataset from Hugging Face
# Option 1: Using git clone (recommended)
git clone https://huggingface.co/datasets/JianhaoDYDY/LikePhys-Benchmark data

# Option 2: Using huggingface-cli
pip install huggingface_hub
huggingface-cli download JianhaoDYDY/LikePhys-Benchmark --repo-type dataset --local-dir ./data
```

2. **Run Single Evaluation**:
```bash
python evaluator.py --model animatediff --data ball_drop --seed 42 --guidance_scale
```

3. **Run Batch Evaluation**:
```bash
bash run_eval.sh
```

### Command Line Arguments

- `--model`: Model to evaluate (e.g., `animatediff`, `cogvideox`, `hunyuan_t2v`, `ltx`, `mochi`)
- `--data`: Physics scenario to test (e.g., `ball_drop`, `ball_collision`, `pendulum`)
- `--seed`: Random seed for reproducibility
- `--guidance_scale`: Use classifier-free guidance (flag)
- `--tag_name`: Custom tag for organizing experiment results

### Sample Scripts

#### Single Model Evaluation
```bash
# Evaluate a single model on one physics scenario
python evaluator.py \
    --model animatediff \
    --data ball_drop \
    --seed 42 \
    --guidance_scale \
    --tag_name "experiment_1"
```

#### Batch Evaluation
```bash
# Run comprehensive evaluation across all models and scenarios
bash run_eval.sh
```

## Dataset

The dataset is hosted on Hugging Face and contains paired videos (physically plausible vs. implausible) across 12 different physics scenarios.

**Download from Hugging Face:** https://huggingface.co/datasets/JianhaoDYDY/LikePhys-Benchmark

```bash
# Option 1: Using git clone (recommended for full dataset)
git clone https://huggingface.co/datasets/JianhaoDYDY/LikePhys-Benchmark data

# Option 2: Using huggingface-cli
pip install huggingface_hub
huggingface-cli download JianhaoDYDY/LikePhys-Benchmark --repo-type dataset --local-dir ./data
```

![LikePhys Dataset Overview](assets/dataset.png)

## Supported Models

- **AnimateDiff** (`animatediff`)
- **AnimateDiff SDXL** (`animatediff_sdxl`)
- **CogVideoX** (`cogvideox`, `cogvideox-5b`)
- **Hunyuan Video** (`hunyuan_t2v`)
- **LTX Video** (`ltx`)
- **ModelScope** (`modelscope`)
- **Wan Video** (`wan2.1-T2V-1.3b`, `wan2.1-T2V-14b`)
- **ZeroScope** (`zeroscope`)

## Results Analysis

After evaluation, use the analysis script to check results

```bash
python read_exp_final.py
```


## Citation

If you use LikePhys in your research, please cite:

```bibtex
@inproceedings{yuan2025likephys,
  title={LikePhys: Evaluating Intuitive Physics Understanding in Video Diffusion Models via Likelihood Preference},
  author={Yuan, Jianhao and Pizzati, Fabio and Pinto, Francesco and Kunze, Lars and Laptev, Ivan and Newman, Paul and Torr, Philip and De Martini, Daniele},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```