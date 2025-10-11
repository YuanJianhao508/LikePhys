# LikePhys: Evaluating intuitive physics understanding in video diffusion models via likelihood preference

## Abstract

Intuitive physics understanding in video diffusion models plays an essential role in building general-purpose physically plausible world simulators, yet accurately evaluating such capacity remains a challenging task due to the difficulty in disentangling physics correctness from visual appearance in generation. To the end, we introduce *LikePhys*, a training-free method that evaluates intuitive physics in video diffusion models by distinguishing physically valid and impossible videos using the denoising objective as an ELBO-based likelihood surrogate on a curated dataset of valid-invalid pairs.

![LikePhys Method Overview](assets/method.png)

## Overview

![LikePhys Results Overview](assets/teaser.png)

We benchmark 9 state-of-the-art video diffusion models across 12 physics scenarios covering rigid body dynamics, fluid dynamics, deformable materials, and optics. Our evaluation metric, Plausibility Preference Error (PPE), demonstrates strong alignment with human preferences and reveals significant variations in physics understanding across different models and physical domains.

## Release Plan

The codebase is temporarily under embargo while we prepare the public release.

- [TODO] Open-source release target: YYYY-MM-DD
- [TODO] License and third‑party notices
- [TODO] Cleaned inference/evaluation scripts
- [TODO] Minimal environment spec and reproducibility notes

In the meantime, please refer to the illustrations above for a high-level
overview of the method and dataset.

<!-- Sample scripts intentionally omitted until public release -->

## Dataset (illustration)

Detailed dataset links and access instructions will be shared upon public release.

![LikePhys Dataset Overview](assets/dataset.png)

## Supported Models

- **AnimateDiff** (`animatediff`)
- **AnimateDiff SDXL** (`animatediff_sdxl`)
- **CogVideoX** (`cogvideox`, `cogvideox-5b`)
- **Hunyuan Video** (`hunyuan_t2v`)
- **LTX Video** (`ltx`, `ltx-0.9.1`, `ltx-0.9.5`)
- **Mochi** (`mochi`)
- **ModelScope** (`modelscope`)
- **Wan Video** (`wan2.1-T2V-1.3b`, `wan2.1-T2V-14b`)
- **ZeroScope** (`zeroscope`)

## Physics Scenarios

The benchmark covers 12 scenarios across 4 physics domains:

### Rigid Body Dynamics
1. **Ball Drop** (`ball_drop`) - Gravity and collision with ground
2. **Ball Collision** (`ball_collision`) - Momentum conservation in collisions
3. **Pendulum** (`pendulum`) - Harmonic motion and energy conservation
4. **Block Slide** (`block_slide`) - Friction and inclined plane dynamics
5. **Pyramid Collapse** (`pyramid`) - Multi-body collision dynamics

### Fluid Dynamics
6. **Fluid Droplet** (`fluid`) - Droplet falling and surface tension
7. **Faucet Flow** (`faucet`) - Continuous fluid flow mechanics
8. **River Flow** (`river`) - Complex fluid dynamics with obstacles

### Deformable Materials
9. **Cloth Drape** (`cloth`) - Soft body physics and draping
10. **Flag Motion** (`flag`) - Cloth-wind interaction

### Optics
11. **Shadow Casting** (`shadow`) - Light source movement and shadow consistency
12. **Camera Motion** (`shadowm`) - Perspective changes and shadow stability

## Repository Contents (temporary)

Only non-code assets are included here prior to the public release:

- `assets/` – figures used in the paper and website
- `README.md` – this page

## Results and Examples

See the project website for qualitative examples and figures.


## Citation

If you use LikePhys in your research, please cite:

```bibtex
TODO
```




