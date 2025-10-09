#!/bin/bash

# Script to download sample videos for the website (4 physics domains)
# Run this after logging in: huggingface-cli login

echo "Downloading sample videos from Hugging Face..."

# Create temp directory
mkdir -p temp_download
cd temp_download

# Download the dataset - 4 domains: Rigid Body, Fluid, Optical, Continuum
huggingface-cli download JianhaoDYDY/LikePhys-Benchmark \
  --repo-type dataset \
  --include "ball_drop_videos/subgroup_000/valid_00.mp4" \
  --include "ball_drop_videos/subgroup_000/over_bounce_00.mp4" \
  --include "faucet_videos/subgroup_000/valid_00.mp4" \
  --include "faucet_videos/subgroup_000/antigravity_fluid_00.mp4" \
  --include "shadow_videos/subgroup_000/valid_00.mp4" \
  --include "shadow_videos/subgroup_000/no_shadow_00.mp4" \
  --include "cloth_drape_videos/subgroup_000/valid_00.mp4" \
  --include "cloth_drape_videos/subgroup_000/floating_cloth_00.mp4" \
  --local-dir ./data

# Copy and rename the videos (4 domains)
echo "Copying videos..."
# Rigid Body Dynamics
cp data/ball_drop_videos/subgroup_000/valid_00.mp4 ../rigid_valid.mp4
cp data/ball_drop_videos/subgroup_000/over_bounce_00.mp4 ../rigid_invalid.mp4

# Fluid Mechanics
cp data/faucet_videos/subgroup_000/valid_00.mp4 ../fluid_valid.mp4
cp data/faucet_videos/subgroup_000/antigravity_fluid_00.mp4 ../fluid_invalid.mp4

# Optical Effects
cp data/shadow_videos/subgroup_000/valid_00.mp4 ../optical_valid.mp4
cp data/shadow_videos/subgroup_000/no_shadow_00.mp4 ../optical_invalid.mp4

# Continuum Mechanics
cp data/cloth_drape_videos/subgroup_000/valid_00.mp4 ../continuum_valid.mp4
cp data/cloth_drape_videos/subgroup_000/floating_cloth_00.mp4 ../continuum_invalid.mp4

# Clean up
cd ..
rm -rf temp_download

echo "✅ Done! Sample videos are ready (4 domains)."
echo ""
echo "To update the website, run:"
echo "  cd /Users/jianhaoyuan/Desktop/project/LikePhys-website"
echo "  git add static/videos/samples/*.mp4"
echo "  git commit -m 'Add sample videos'"
echo "  git push origin gh-pages"

