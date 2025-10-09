# Video Samples for LikePhys Project Page

This directory contains sample video clips displayed on the project website, organized by the 4 physics domains.

## Required Videos (8 total - 4 domains × 2 videos each):

1. **rigid_valid.mp4** - Valid rigid body dynamics (e.g., ball drop with normal gravity)
2. **rigid_invalid.mp4** - Rigid body with violation (e.g., over-bouncing ball)
3. **fluid_valid.mp4** - Valid fluid mechanics (e.g., faucet flow)
4. **fluid_invalid.mp4** - Fluid with violation (e.g., anti-gravity fluid)
5. **optical_valid.mp4** - Valid optical effects (e.g., correct shadow casting)
6. **optical_invalid.mp4** - Optical with violation (e.g., missing shadow)
7. **continuum_valid.mp4** - Valid continuum mechanics (e.g., cloth draping)
8. **continuum_invalid.mp4** - Continuum with violation (e.g., floating cloth)

## How to Add Videos:

### Download the dataset first:
```bash
# Download from Hugging Face
huggingface-cli download JianhaoDYDY/LikePhys-Benchmark --repo-type dataset --local-dir ./temp_data
```

### From the downloaded data folder:
```bash
# Example: Copy valid and invalid samples from your dataset
cp ./temp_data/ball_drop_videos/subgroup_000/valid_00.mp4 rigid_valid.mp4
cp ./temp_data/ball_drop_videos/subgroup_000/over_bounce_00.mp4 rigid_invalid.mp4
cp ./temp_data/faucet_videos/subgroup_000/valid_00.mp4 fluid_valid.mp4
cp ./temp_data/faucet_videos/subgroup_000/antigravity_fluid_00.mp4 fluid_invalid.mp4
cp ./temp_data/shadow_videos/subgroup_000/valid_00.mp4 optical_valid.mp4
cp ./temp_data/shadow_videos/subgroup_000/no_shadow_00.mp4 optical_invalid.mp4
cp ./temp_data/cloth_drape_videos/subgroup_000/valid_00.mp4 continuum_valid.mp4
cp ./temp_data/cloth_drape_videos/subgroup_000/floating_cloth_00.mp4 continuum_invalid.mp4
```

### Tips:
- Keep videos short (3-5 seconds) for fast loading
- Compress if needed: `ffmpeg -i input.mp4 -vcodec h264 -acodec aac -vf scale=640:-1 output.mp4`
- Max file size recommendation: ~2MB per video
- Use the provided `download_samples.sh` script for automatic download and setup

## Quick Setup:

Just run the provided script:
```bash
./download_samples.sh
```

This will automatically download and rename all 8 required videos from the Hugging Face dataset.