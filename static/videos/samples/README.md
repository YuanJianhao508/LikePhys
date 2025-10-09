# Video Samples for Website

Add your sample video files here with the following names:

## Required Video Files:

1. **ball_drop_valid.mp4** - Valid ball drop scenario showing normal physics
2. **ball_drop_invalid.mp4** - Ball drop with violation (e.g., over-bouncing, penetration)
3. **pendulum_valid.mp4** - Valid pendulum motion
4. **pendulum_invalid.mp4** - Pendulum with violation (e.g., reverse gravity, varying frequency)
5. **fluid_valid.mp4** - Valid fluid flow scenario
6. **fluid_invalid.mp4** - Fluid with violation (e.g., anti-gravity, teleportation)

## How to Add Videos:

### Download the dataset first:
```bash
# Download from Hugging Face
huggingface-cli download JianhaoDYDY/LikePhys-Benchmark --repo-type dataset --local-dir ./temp_data
```

### From the downloaded data folder:
```bash
# Example: Copy valid and invalid samples from your dataset
cp ./temp_data/ball_drop_videos/subgroup_000/valid_00.mp4 ball_drop_valid.mp4
cp ./temp_data/ball_drop_videos/subgroup_000/over_bounce_00.mp4 ball_drop_invalid.mp4
cp ./temp_data/pendulum_videos/subgroup_000/valid_00.mp4 pendulum_valid.mp4
cp ./temp_data/pendulum_videos/subgroup_000/reverse_gravity_00.mp4 pendulum_invalid.mp4
cp ./temp_data/fluid_videos/subgroup_000/valid_00.mp4 fluid_valid.mp4
cp ./temp_data/fluid_videos/subgroup_000/antigravity_fluid_00.mp4 fluid_invalid.mp4
```

### Tips:
- Keep videos short (3-5 seconds) for fast loading
- Compress if needed: `ffmpeg -i input.mp4 -vcodec h264 -acodec aac -vf scale=640:-1 output.mp4`
- Max file size recommendation: ~2MB per video
- Format: MP4 (H.264)

### After adding videos:
```bash
cd /Users/jianhaoyuan/Desktop/project/LikePhys-website
git add static/videos/samples/*.mp4
git commit -m "Add sample videos to website"
git push origin gh-pages
```

