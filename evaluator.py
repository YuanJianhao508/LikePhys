import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import cv2
import numpy as np
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKLWan, HunyuanVideoTransformer3DModel
from diffusers import DPMSolverMultistepScheduler, DDIMScheduler, MotionAdapter
from pipeline.svd_pipeline import StableVideoDiffusionPipeline, retrieve_timesteps, _append_dims
from pipeline.animatediff_pipeline import AnimateDiffPipeline
from pipeline.animatediffsdxl_pipeline import AnimateDiffSDXLPipeline
# from pipeline.cogvideox_pipeline import CogVideoXPipeline
from pipeline.cogvideox_new_pipeline import CogVideoXPipeline
# from pipeline.zeroscope_pipeline import TextToVideoSDPipeline
from pipeline.modelscope_pipeline import TextToVideoSDPipeline
from pipeline.wan_pipeline import WanVideoToVideoPipeline
from pipeline.hunyuan_i2v_pipeline import HunyuanVideoImageToVideoPipeline
from pipeline.hunyuan_t2v_pipeline import HunyuanVideoPipeline, DEFAULT_PROMPT_TEMPLATE
from pipeline.mochi_pipeline import MochiPipeline
from pipeline.ltx_pipeline import LTXPipeline
# from pipeline.cosmos_pipeline import CosmosTextToWorldPipeline
from scheduler.euler_discrete import EulerDiscreteScheduler
from scheduler.unipc_multistep import UniPCMultistepScheduler
from accelerate import Accelerator
from tqdm.auto import tqdm
from diffusers.utils import export_to_video, load_image, load_video
from diffusers.utils.torch_utils import randn_tensor
import json
import argparse

def get_prompt(args):
    if args.data == "ball_drop":
        prompt = "ball dropping and colliding with the ground, in empty background"
        # negative_prompt = "violate physics"
    elif args.data == "ball_collision":
        prompt = "two balls colliding with each other"
        # negative_prompt = "violate physics"
    elif args.data == "pendulum":
        prompt = "a pendulum swinging"
        # negative_prompt = "violate physics"
    elif args.data == "block_slide":
        prompt = "a block sliding on a slope"
        # negative_prompt = "violate physics"
    elif args.data == "fluid":
        prompt = "a droplet falling"
        # negative_prompt = "violate physics"
    elif args.data == "faucet":
        prompt = "fluid flowing from a faucet"
        # negative_prompt = "violate physics"
    elif args.data == "cloth":
        prompt = "a piece of cloth dropping to the obstacle on the ground"
        # negative_prompt = "violate physics"
    elif args.data == "flag":
        prompt = "a piece of cloth waving in the wind"
        # negative_prompt = "violate physics"
    elif args.data == "river":
        prompt = "fluid flowing in a tank with obstacles"
        # negative_prompt = "violate physics"
    elif args.data == "shadow":
        prompt = "light source moving around an object showing its shadow"
        # negative_prompt = "violate physics"
    elif args.data == "pyramid":
        prompt = "a cube crash into a pile of spheres"
        # negative_prompt = "violate physics"
    elif args.data == "shadowm":
        prompt = "camera moving around an object"
        # negative_prompt = "violate physics"
    elif args.data == 'sample':
        prompt = 'two balls colliding with each other'
    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

    return prompt, negative_prompt




def load_video_and_first_frame(pipe, video_path, height, width, num_frames):
    """
    Load video via `load_video`, resize each frame to (width, height), 
    evenly sample `num_frames` frames (duplicates if needed), 
    preprocess to a tensor and return it along with the first frame.
    """
    # define a convert_method that resizes every frame
    def _resize_frames(frames):
        return [frame.convert("RGB").resize((width, height)) for frame in frames]

    all_frames = load_video(video_path, convert_method=_resize_frames)
    # keep the (already resized) first frame
    first_frame = all_frames[0]

    total = len(all_frames)
    idxs = np.linspace(0, total - 1, num_frames)
    idxs = np.round(idxs).astype(int).tolist()
    sampled = [all_frames[i] for i in idxs]

    # preprocess into a batch tensor
    video_tensor = pipe.video_processor.preprocess(sampled, height=height, width=width)
    video_tensor = video_tensor.unsqueeze(0)  # [1, C, F, H, W]


    return video_tensor, first_frame


@torch.no_grad()
def evaluate_video(args, video_path, pipe, noise_aug_strength=0.02, num_videos_per_prompt=1):
    """
    Evaluate a single video:
      1. Load video and get video_tensor and first frame (image condition).
      2. Convert video_tensor to VAE latent representation.
      3. Preprocess first frame, add noise, and encode through VAE to get image_latents.
      4. Add noise to video latent, predict noise through unet, and calculate weighted MSE between predicted and real noise.
    Returns: loss and detailed log information (dictionary).
    """
    device = pipe._execution_device
    generator = set_seed(args.subgroup_seed)    
    # 1. Load video and first frame (with uniform resize)
    video_tensor, first_frame = load_video_and_first_frame(pipe, video_path, height=args.height, width=args.width, num_frames=args.num_frames)
    if video_tensor is None:
        print(f"Failed to load video: {video_path}")
        return None, None

    # 2. Convert to VAE latent representation
    video_tensor = video_tensor.to(device, dtype=pipe.vae.dtype)
    sample = pipe.tensor_to_vae_latent(video_tensor)  # shape: [B, latent_channels, frames, ...]
    bsz = sample.shape[0]

    # IMAGE to VIDEO
    if args.model == "svd":
        # 3. Image condition processing: preprocess first frame and add noise
        conditioning_image = pipe.video_processor.preprocess(first_frame, height=args.height, width=args.width).to(device)
        noise = randn_tensor(conditioning_image.shape, generator=generator, device=device, dtype=conditioning_image.dtype)
        conditioning_image = conditioning_image + noise_aug_strength * noise

        # 4. Encode through VAE to get image_latents
        needs_upcasting = pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast
        if needs_upcasting:
            pipe.vae.to(dtype=torch.float32)
        image_latents = pipe._encode_vae_image(
            conditioning_image,
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        )
        if needs_upcasting:
            pipe.vae.to(dtype=torch.float16)
        # Repeat each frame to match video latent frame count (assuming pipe.unet.config.num_frames exists)
        num_frames = pipe.unet.config.num_frames
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

        image_embeddings = pipe._encode_image(first_frame, device, num_videos_per_prompt, pipe.do_classifier_free_guidance)
        added_time_ids = pipe._get_add_time_ids(args.fps - 1, 127, noise_aug_strength, image_latents.dtype, bsz, num_videos_per_prompt, pipe.do_classifier_free_guidance)
        added_time_ids = added_time_ids.to(device)

    # TEXT to VIDEO
    elif args.model == "animatediff":
        # 3. Get prompt embeddings
        prompt, negative_prompt = get_prompt(args)
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt,
            device,
            1,
            pipe.do_classifier_free_guidance,
            negative_prompt,
        )
        if pipe.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        prompt_embeds = prompt_embeds.repeat_interleave(repeats=args.num_frames, dim=0)
    
    elif args.model == "animatediff_sdxl":
        prompt, negative_prompt = get_prompt(args)
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=device,
            num_videos_per_prompt=1,
            do_classifier_free_guidance=pipe.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=None,
        )

        # 7. Prepare added time ids & embeddings
        original_size = (args.height, args.width)
        target_size = (args.height, args.width)
        crops_coords_top_left = (0, 0)
        add_text_embeds = pooled_prompt_embeds
        if pipe.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim

        add_time_ids = pipe._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        negative_add_time_ids = add_time_ids

        if pipe.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.repeat_interleave(repeats=args.num_frames, dim=0)
        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(bsz * num_videos_per_prompt, 1)

        timestep_cond = None
        if pipe.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(pipe.guidance_scale - 1).repeat(bsz * num_videos_per_prompt)
            timestep_cond = pipe.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=pipe.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=sample.dtype)


    elif args.model in ["cogvideox", "cogvideox-5b", "cogvideox1.5-5b"]:
        do_classifier_free_guidance = pipe._guidance_scale > 1.0
        pipe.do_classifier_free_guidance = do_classifier_free_guidance
        prompt, negative_prompt = get_prompt(args)
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=226,
            device=device,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        image_rotary_emb = (
            pipe._prepare_rotary_positional_embeddings(args.height, args.width, sample.size(1), device)
            if pipe.transformer.config.use_rotary_positional_embeddings
            else None
        )
    
    elif args.model in ["zeroscope", "modelscope"]:
        do_classifier_free_guidance = pipe._guidance_scale > 1.0
        pipe.do_classifier_free_guidance = do_classifier_free_guidance
        prompt, negative_prompt = get_prompt(args)
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt,
            device,
            1,
            pipe.do_classifier_free_guidance,
            negative_prompt,
        )
        if pipe.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
    
    elif args.model in ["wan2.1-T2V-1.3b", "wan2.1-T2V-14b"]:
        # do_classifier_free_guidance = pipe._guidance_scale > 1.0
        # pipe.do_classifier_free_guidance = do_classifier_free_guidance
        # print("Here:", pipe.do_classifier_free_guidance)
        prompt, negative_prompt = get_prompt(args)
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=pipe.do_classifier_free_guidance,
            num_videos_per_prompt=1,
            max_sequence_length=512,
            device=device,
        )
        transformer_dtype = pipe.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

    elif args.model in ["hunyuan_i2v", "hunyuan_t2v"]:
        prompt, negative_prompt = get_prompt(args)
        prompt_2 = None
        transformer_dtype = pipe.transformer.dtype
        if args.model == "hunyuan_i2v":
            vae_dtype = pipe.vae.dtype
            image_tensor = pipe.video_processor.preprocess(first_frame, height=args.height, width=args.width).to(device, vae_dtype)
            prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = pipe.encode_prompt(
                image=image_tensor,
                prompt=prompt,
                prompt_2=prompt_2,
                prompt_template=DEFAULT_PROMPT_TEMPLATE,
                num_videos_per_prompt=1,
                device=device,
                max_sequence_length=256,
            )
        elif args.model == "hunyuan_t2v":
            prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = pipe.encode_prompt(
                prompt=prompt,
                prompt_2=prompt_2,
                prompt_template=DEFAULT_PROMPT_TEMPLATE,
                num_videos_per_prompt=1,
                device=device,
                max_sequence_length=256,
            )
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        prompt_attention_mask = prompt_attention_mask.to(transformer_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(transformer_dtype)

    elif args.model in ["ltx","ltx-0.9.5","ltx-0.9.1","ltx-0.9.7"]:
        # do_classifier_free_guidance = pipe._guidance_scale > 1.0
        # pipe.do_classifier_free_guidance = do_classifier_free_guidance
        prompt, negative_prompt = get_prompt(args)
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=pipe.do_classifier_free_guidance,
            num_videos_per_prompt=1,
            max_sequence_length=128,
            device=device,
        )
        if pipe.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

    elif args.model == "mochi":
        prompt, negative_prompt = get_prompt(args)
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=pipe.do_classifier_free_guidance,
            num_videos_per_prompt=1,
            max_sequence_length=256,
            device=device,
        )
        if pipe.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

    elif model_type == "cosmos":
        (
            prompt_embeds,
            negative_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=pipe.do_classifier_free_guidance,
            num_videos_per_prompt=1,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            device=device,
            max_sequence_length=512,
        )

    loss_array = np.array([])
    total_ntrail = args.timestep_num
    # if args.timestep_strategy == "uniform":
    t_sample_step = pipe.scheduler.timesteps.shape[0] / total_ntrail

    for nt in range(total_ntrail):
        # 5. Add noise to video latent
        # timestep_id = 500
        
        timestep_id = int(nt * t_sample_step)
        sigmas = pipe.scheduler.sigmas[timestep_id]
        timesteps = pipe.scheduler.timesteps[timestep_id]
        # Generate random noise with the same shape and data type as sample
        noise = randn_tensor(sample.shape, generator=generator, device=device, dtype=sample.dtype)
        # print(noise.mean().item())
        # Add noise to the original latent using the customize_add_noise method
        noised_sample = pipe.scheduler.customize_add_noise(sample, noise, timesteps)
        

        if args.model == "svd":
            latent_model_input = torch.cat([noised_sample] * 2) if pipe.do_classifier_free_guidance else noised_sample
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timesteps)
                # Concatenate image_latents over channels dimension
            latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

            # 7. Get noise prediction through unet.
            model_output = pipe.unet(
                latent_model_input,
                timesteps,
                encoder_hidden_states=image_embeddings,
                added_time_ids=added_time_ids,
                return_dict=False,
            )[0]

        elif args.model == "animatediff":
            latent_model_input = torch.cat([noised_sample] * 2) if pipe.do_classifier_free_guidance else noised_sample
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timesteps)
            model_output = pipe.unet(
                latent_model_input,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=None,
                added_cond_kwargs=None,
            ).sample

        elif args.model == "animatediff_sdxl":
            timesteps = timesteps.to(device)
            latent_model_input = torch.cat([noised_sample] * 2) if pipe.do_classifier_free_guidance else noised_sample

            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timesteps)

            # predict the noise residual
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

            model_output = pipe.unet(
                latent_model_input,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=None,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
        
        elif args.model in ["cogvideox", "cogvideox-5b", "cogvideox1.5-5b"]:
            latent_model_input = torch.cat([noised_sample] * 2) if do_classifier_free_guidance else noised_sample
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timesteps)
            timestep = timesteps.expand(latent_model_input.shape[0])
            model_output = pipe.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=None,
                return_dict=False,
            )[0]
            model_output = model_output.float()

        elif args.model in ["zeroscope", "modelscope"]:
            latent_model_input = torch.cat([noised_sample] * 2) if pipe.do_classifier_free_guidance else noised_sample
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timesteps)
            model_output = pipe.unet(
                latent_model_input,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]

        elif args.model in ["wan2.1-T2V-1.3b", "wan2.1-T2V-14b"]:
            latent_model_input = noised_sample.to(transformer_dtype)
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timesteps)
            timestep = timesteps.expand(noised_sample.shape[0]).to(device)
            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                attention_kwargs=None,
                return_dict=False,
            )[0]

            if pipe.do_classifier_free_guidance:
                noise_uncond = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    attention_kwargs=None,
                    return_dict=False,
                )[0]
                noise_pred = noise_uncond + pipe._guidance_scale * (noise_pred - noise_uncond)

            model_output = noise_pred

        elif args.model in ["hunyuan_i2v", "hunyuan_t2v"]:
            pipe.do_classifier_free_guidance = False
            transformer_dtype = pipe.transformer.dtype
            guidance = torch.tensor([pipe._guidance_scale] * noised_sample.shape[0], dtype=transformer_dtype, device=device) * 1000.0
            latent_model_input = noised_sample.to(transformer_dtype)
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timesteps)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = timesteps.expand(noised_sample.shape[0]).to(noised_sample.dtype).to(device)

            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                pooled_projections=pooled_prompt_embeds,
                guidance=guidance,
                attention_kwargs=None,
                return_dict=False,
            )[0]
            model_output = noise_pred
        
        elif args.model in ["ltx","ltx-0.9.5","ltx-0.9.1","ltx-0.9.7"]:
            num_frames = args.num_frames
            latent_num_frames = (num_frames - 1) // pipe.vae_temporal_compression_ratio + 1
            latent_height = args.height // pipe.vae_spatial_compression_ratio
            latent_width = args.width // pipe.vae_spatial_compression_ratio
            video_sequence_length = latent_num_frames * latent_height * latent_width
            rope_interpolation_scale = (
                pipe.vae_temporal_compression_ratio / 25,
                pipe.vae_spatial_compression_ratio,
                pipe.vae_spatial_compression_ratio,
            )
            latent_model_input = torch.cat([noised_sample] * 2) if pipe.do_classifier_free_guidance else noised_sample
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timesteps)
            latent_model_input = latent_model_input.to(prompt_embeds.dtype)

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = timesteps.expand(latent_model_input.shape[0]).to(device)

            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                encoder_attention_mask=prompt_attention_mask,
                num_frames=latent_num_frames,
                height=latent_height,
                width=latent_width,
                rope_interpolation_scale=rope_interpolation_scale,
                return_dict=False,
            )[0]
            noise_pred = noise_pred.float()
            model_output = noise_pred

        elif args.model == "mochi":
            latent_model_input = torch.cat([noised_sample] * 2) if pipe.do_classifier_free_guidance else noised_sample
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = timesteps.expand(latent_model_input.shape[0]).to(noised_sample.dtype)

            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                encoder_attention_mask=prompt_attention_mask,
                attention_kwargs=None,
                return_dict=False,
            )[0]
            # Mochi CFG + Sampling runs in FP32
            model_output = noise_pred.to(torch.float32)

        elif args.model == "cosmos":
            transformer_dtype = pipe.transformer.dtype
            padding_mask = latents.new_zeros(1, 1, args.height, args.width, dtype=transformer_dtype)
            latent_model_input = latents
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timesteps)
            latent_model_input = latent_model_input.to(transformer_dtype)

            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                fps=args.fps,
                padding_mask=padding_mask,
                return_dict=False,
            )[0]

            sample = latents
            if pipe.do_classifier_free_guidance:
                noise_pred_uncond = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    fps=args.fps,
                    padding_mask=padding_mask,
                    return_dict=False,
                )[0]
                model_output = torch.cat([noise_pred_uncond, noise_pred])
                sample = torch.cat([sample, sample])

        if pipe.do_classifier_free_guidance and args.model not in ["wan2.1-T2V-1.3b", "wan2.1-T2V-14b", "hunyuan_i2v", "hunyuan_t2v", "cosmos"]:

            noise_pred_uncond, noise_pred_cond = model_output.chunk(2)
            model_output = noise_pred_uncond + pipe._guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # 8. Calculate noise prediction based on prediction_type. Here, we use "v_prediction" as an example.
        # print("Here!!!!!!!!!!!",pipe.scheduler.config.prediction_type)
        if pipe.scheduler.config.prediction_type == "v_prediction":
            pred_sample = model_output * (-sigmas / (sigmas**2 + 1)**0.5) + (sample / (sigmas**2 + 1))
            noise_pred = (sample - pred_sample) / sigmas
        elif pipe.scheduler.config.prediction_type == "epsilon":
            pred_sample = sample - sigmas * model_output
            noise_pred = model_output
        elif pipe.scheduler.config.prediction_type == "flow_prediction":
            # model_output = flow_pred = ε_pred - x_pred
            # model_output = sample - sigmas * model_output
            # noise_pred = model_output
            x0_pred = sample - sigmas * model_output
            alpha       = (sigmas**2 + 1) ** 0.5
            # recover the noise prediction ε_pred
            noise_pred  = (alpha * model_output + sample) / (alpha + sigmas)
            # then predicted sample component (α * x_pred)
            pred_sample = sample - sigmas * noise_pred
        else:
            pass


        # 9. Calculate weighted MSE loss (loss) as a metric for video physical plausibility
        loss = F.mse_loss(noise, noise_pred, reduction='mean').item()
        loss_array = np.append(loss_array, loss)
    
    loss = np.mean(loss_array)
    
    # print(loss)
    log_info = {
        "loss": loss,
        "noise_pred_mean": noise_pred.mean().item(),
        "true_noise_mean": noise.mean().item(),
        "loss_array": list(loss_array)
    }

    if USE_WANDB:
        wandb.log(log_info)
    else:
        print(f"Video: {video_path}, Loss: {loss:.4f}, Noise pred: {log_info['noise_pred_mean']:.4f}, True noise: {log_info['true_noise_mean']:.4f}")

    visualize = False
    if visualize:
        with torch.no_grad():
            if args.model == "svd":
                latents = pipe.scheduler.step(noise_pred, timesteps, noised_sample, return_dict=False)[0]
                video = pipe.decode_latents(latents, 14, 7)
                video = pipe.video_processor.postprocess_video(video=video)[0]
               
            elif args.model in ["zeroscope", "modelscope"]:
                # reshape latents
                bsz, channel, frames, width, height = noised_sample.shape
                noised_sample = noised_sample.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)
                noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)
                latents = pipe.scheduler.step(noise_pred, timesteps, noised_sample)[0]
                latents = latents[None, :].reshape(bsz, frames, channel, width, height).permute(0, 2, 1, 3, 4)
                video = pipe.decode_latents(latents)
                video = pipe.video_processor.postprocess_video(video=video)[0]
                # import pdb; pdb.set_trace()
            elif args.model in ["wan2.1-T2V-1.3b", "wan2.1-T2V-14b"]:
                # latents = noised_sample.to(pipe.vae.dtype)
                pipe.scheduler.config.prediction_type = "epsilon"
                latents = pipe.scheduler.step(model_output, timesteps, noised_sample, return_dict=False)[0]
                latents_mean = (
                    torch.tensor(pipe.vae.config.latents_mean)
                    .view(1, pipe.vae.config.z_dim, 1, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(
                    latents.device, latents.dtype
                )
                latents = latents / latents_std + latents_mean
                latents = latents.to(dtype=torch.bfloat16)
                video = pipe.vae.decode(latents, return_dict=False)[0]
                video = pipe.video_processor.postprocess_video(video)[0]
            elif args.model in ["hunyuan_i2v", "hunyuan_t2v"]:
                latents = noised_sample.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor
                video = pipe.vae.decode(latents, return_dict=False)[0]
                video = pipe.video_processor.postprocess_video(video, output_type="np")[0]
            elif args.model in ["ltx","ltx-0.9.5","ltx-0.9.1","ltx-0.9.7"]:
                latents = pipe.scheduler.step(noise_pred, timesteps, noised_sample, return_dict=False)[0]
                latents = pipe._unpack_latents(
                    latents,
                    latent_num_frames,
                    latent_height,
                    latent_width,
                    pipe.transformer_spatial_patch_size,
                    pipe.transformer_temporal_patch_size,
                )
                latents = pipe._denormalize_latents(
                    latents, pipe.vae.latents_mean, pipe.vae.latents_std, pipe.vae.config.scaling_factor
                )
                latents = latents.to(prompt_embeds.dtype)
                video = pipe.vae.decode(latents, timestep, return_dict=False)[0]
                video = pipe.video_processor.postprocess_video(video, output_type="np")[0]
            elif args.model == "mochi":
                latents_dtype = noised_sample.dtype
                latents = pipe.scheduler.step(noise_pred, timesteps, noised_sample, return_dict=False)[0]
                latents = noised_sample.to(latents_dtype)
                has_latents_mean = hasattr(pipe.vae.config, "latents_mean") and pipe.vae.config.latents_mean is not None
                has_latents_std = hasattr(pipe.vae.config, "latents_std") and pipe.vae.config.latents_std is not None
                if has_latents_mean and has_latents_std:
                    latents_mean = (
                        torch.tensor(pipe.vae.config.latents_mean).view(1, 12, 1, 1, 1).to(latents.device, latents.dtype)
                    )
                    latents_std = (
                        torch.tensor(pipe.vae.config.latents_std).view(1, 12, 1, 1, 1).to(latents.device, latents.dtype)
                    )
                    latents = latents * latents_std / pipe.vae.config.scaling_factor + latents_mean
                else:
                    latents = latents / pipe.vae.config.scaling_factor

                video = pipe.vae.decode(latents, return_dict=False)[0]
                
            else:
                latents = pipe.scheduler.step(noise_pred, timesteps, noised_sample, return_dict=False)[0]
                # latents = latents[:, 0:]
                video = pipe.decode_latents(latents.to(dtype=pipe.vae.dtype))
                video = pipe.video_processor.postprocess_video(video=video)[0]
            export_to_video(video, f"./temp/check_video.mp4", fps=16)

    return loss, log_info

def evaluate_dataset(args, dataset_dir, pipe):
    """
    Evaluate videos grouped by subgroups.
    Store per-video losses without averaging.
    """
    results = {}
    
    for sub_idx, subgroup_id in enumerate(sorted(os.listdir(dataset_dir))):
        subgroup_path = os.path.join(dataset_dir, subgroup_id)
        if not os.path.isdir(subgroup_path):
            continue

        subgroup_seed = args.seed + sub_idx
        args.subgroup_seed = subgroup_seed

        subgroup_results = {}
        for video_name in os.listdir(subgroup_path):
            if not video_name.endswith('.mp4'):
                continue
            
            video_path = os.path.join(subgroup_path, video_name)
            variation_type = video_name.rsplit('_', 1)[0]  # e.g., over_bounce_00.mp4 → over_bounce
            loss, log_info = evaluate_video(args, video_path, pipe)
            if loss is not None:
                if variation_type not in subgroup_results:
                    subgroup_results[variation_type] = {}
                subgroup_results[variation_type][video_name] = {
                    "loss": loss,
                    "noise_pred_mean": log_info["noise_pred_mean"],
                    "true_noise_mean": log_info["true_noise_mean"],
                    "loss_array": log_info["loss_array"]
                }

        if subgroup_results:
            results[subgroup_id] = subgroup_results

    return results


def compute_misrank_normalized(results):
    """
    Compute mis-rank within each subgroup (valid vs invalid losses).
    Automatically discovers all variation_types ≠ "valid" as invalid types.
    """
    # 1) discover all invalid variation types across all subgroups
    invalid_types = sorted({
        var_type
        for subgroup_data in results.values()
        for var_type in subgroup_data.keys()
        if var_type != "valid"
    })
    print(f"Discovered invalid types: {invalid_types}")

    # 2) compute mis-rank for each invalid type
    misrank_results = {}
    for var_type in invalid_types:
        subgroup_misranks = []
        total_pairs = 0

        for subgroup_id, subgroup_data in results.items():
            # need at least one "valid" and one of this invalid type
            if "valid" not in subgroup_data or var_type not in subgroup_data:
                continue

            valid_losses = [info["loss"] for info in subgroup_data["valid"].values()]
            invalid_losses = [info["loss"] for info in subgroup_data[var_type].values()]

            # form all valid–invalid pairs
            pairs = [(v, i) for v in valid_losses for i in invalid_losses]
            if not pairs:
                continue

            misrank = sum(1 for v, i in pairs if v > i)
            misrank_ratio = misrank / len(pairs)

            subgroup_misranks.append(misrank_ratio)
            total_pairs += len(pairs)

        avg_misrank = float(np.mean(subgroup_misranks)) if subgroup_misranks else 0.0

        misrank_results[var_type] = {
            "misrank_ratio": avg_misrank,
            "total_pairs": total_pairs,
            "subgroup_misranks": subgroup_misranks,
        }

    return misrank_results

def set_seed(seed):
    # Python built-in random
    random.seed(seed)
    # NumPy random
    np.random.seed(seed)
    # PyTorch random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Torch Generator
    gen = torch.Generator()
    gen.manual_seed(seed)
    
    return gen



def get_model_params(model_type):
    """Get model-specific parameters based on model type"""
    params = {}
    if model_type == "svd":
        params["height"] = 224
        params["width"] = 224
        params["fps"] = 7  # Note: In original Stable Video Diffusion implementation, fps condition is fps-1
        params["num_frames"] = 14
    elif model_type == "animatediff":
        params["height"] = 512
        params["width"] = 512
        params["fps"] = 16
        params["num_frames"] = 16
    elif model_type == "animatediff_sdxl":
        params["height"] = 1024
        params["width"] = 1024
        params["fps"] = 16
        params["num_frames"] = 16
    elif "cogvideox" in model_type:
        if "1.5" in model_type:
            params["height"] = 768
            params["width"] = 1360
            params["fps"] = 16
            params["num_frames"] = 85 # 81 as default but pad 4 for 1.5
        else:
            params["height"] = 480
            params["width"] = 720
            params["fps"] = 16
            params["num_frames"] = 49
    elif model_type in ["zeroscope", "modelscope"]:
        params["height"] = 320
        params["width"] = 576
        params["fps"] = 16
        params["num_frames"] = 24
    elif model_type in ["wan2.1-T2V-1.3b", "wan2.1-T2V-14b"]:
        params["height"] = 480
        params["width"] = 832
        params["fps"] = 16
        # params["num_frames"] = 81
        params["num_frames"] = 33
    elif model_type in ["hunyuan_i2v", "hunyuan_t2v"]:
        params["height"] = 320
        params["width"] = 512
        params["fps"] = 16
        params["num_frames"] = 61
    elif model_type in ["ltx","ltx-0.9.5","ltx-0.9.1","ltx-0.9.7"]:
        params["height"] = 480
        params["width"] = 704
        params["fps"] = 25
        params["num_frames"] = 161
    elif model_type == "mochi":
        params["height"] = 480
        params["width"] = 848
        params["fps"] = 16
        params["num_frames"] = 85
    elif model_type == "cosmos":
        params["height"] = 704
        params["width"] = 1280
        params["fps"] = 30
        params["num_frames"] = 121
    else:
        raise ValueError(f"Invalid model: {model_type}")
    return params

def initialize_model(args):
    """Initialize model-specific pipeline"""
    model_type = args.model
    use_cfg = args.guidance_scale
    if model_type == "svd":
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid", torch_dtype=torch.float32, variant="fp16", cache_dir="./cache"
        )
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        pipe._guidance_scale = 1.0 if not use_cfg else 3.0
    elif model_type in ["animatediff"]:
        adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16, cache_dir="./cache")
        model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
        pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=torch.float16, cache_dir="./cache")
        scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler = scheduler
        pipe._guidance_scale = 1.0 if not use_cfg else 7.5
        pipe.enable_vae_slicing()
        pipe.enable_model_cpu_offload()
    elif model_type in ["animatediff_sdxl"]:
        adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-sdxl-beta", torch_dtype=torch.float16, cache_dir="./cache")
        # adapter = MotionAdapter.from_pretrained("a-r-r-o-w/animatediff-motion-adapter-sdxl-beta", torch_dtype=torch.float16, cache_dir="./cache")
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        
        pipe = AnimateDiffSDXLPipeline.from_pretrained(
            model_id,
            motion_adapter=adapter,
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir="./cache"
        ).to("cuda")   
        scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler = scheduler
        pipe._guidance_scale = 1.0 if not use_cfg else 7.5
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
    elif model_type in ["cogvideox", "cogvideox-5b"]:    
        if model_type == "cogvideox":
            model_path = "THUDM/CogVideoX-2b"
        elif model_type == "cogvideox-5b":
            model_path = "THUDM/CogVideoX-5B"
        # Load the motion adapter
        pipe = CogVideoXPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            cache_dir="./cache"
        )
        scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler = scheduler
        pipe._guidance_scale = 1.0 if not use_cfg else 6.0
        # enable memory savings
        pipe.enable_model_cpu_offload()
        pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    elif model_type in ["cogvideox1.5-5b"]:
        pipe = CogVideoXPipeline.from_pretrained(
            "THUDM/CogVideoX1.5-5B",
            torch_dtype=torch.bfloat16,
            cache_dir="./cache"
        )
        scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler = scheduler
        pipe._guidance_scale = 1.0 if not use_cfg else 6.0
        # pipe.enable_model_cpu_offload()
        pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
    elif model_type in ["zeroscope", "modelscope"]:
        if model_type == "zeroscope":
            model_path = "cerspense/zeroscope_v2_576w"
        elif model_type == "modelscope":
            model_path = "damo-vilab/text-to-video-ms-1.7b"
        pipe = TextToVideoSDPipeline.from_pretrained(model_path, torch_dtype=torch.float16, cache_dir="./cache")
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe._guidance_scale = 1.0 if not use_cfg else 5.0
        pipe.enable_model_cpu_offload()
    elif model_type in ["wan2.1-T2V-1.3b", "wan2.1-T2V-14b"]:
        model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" if model_type == "wan2.1-T2V-1.3b" else "Wan-AI/Wan2.1-T2V-14B-Diffusers"
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        pipe = WanVideoToVideoPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        # pipe.vae.enable_tiling()
        # pipe.vae.enable_slicing()
        # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        # pipe.to("cuda")
        pipe._guidance_scale = 1.0 if not use_cfg else 5.0
    elif model_type in ["hunyuan_i2v", "hunyuan_t2v"]:
        model_id = "hunyuanvideo-community/HunyuanVideo" if model_type == "hunyuan_t2v" else "hunyuanvideo-community/HunyuanVideo-I2V"
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.bfloat16, cache_dir="./cache"
        )
        if model_type == "hunyuan_i2v":
            pipe = HunyuanVideoImageToVideoPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.float16, cache_dir="./cache")
        else:
            pipe = HunyuanVideoPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.float16, cache_dir="./cache")
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.vae.enable_tiling()
        pipe.to("cuda")
        pipe._guidance_scale = 1.0 if not use_cfg else 6.0
    elif model_type in ["ltx","ltx-0.9.5","ltx-0.9.1","ltx-0.9.7"]:
        if model_type == "ltx":
            model_id = "Lightricks/LTX-Video"
        elif model_type == "ltx-0.9.5":
            model_id= "Lightricks/LTX-Video-0.9.5"
        elif model_type == "ltx-0.9.1":
            model_id= "Lightricks/LTX-Video-0.9.1"
        elif model_type == "ltx-0.9.7":
            model_id= "LTX-Video-0.9.7-distilled"
        pipe = LTXPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, cache_dir="./cache")
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to("cuda")
        pipe._guidance_scale = 1.0 if not use_cfg else 3.0
    elif model_type == "mochi":
        pipe = MochiPipeline.from_pretrained("genmo/mochi-1-preview", variant="bf16", torch_dtype=torch.bfloat16, cache_dir="./cache")
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        # reduce memory requirements
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_tiling()
        pipe._guidance_scale = 1.0 if not use_cfg else 4.5
    elif model_type == "cosmos":
        model_id = "nvidia/Cosmos-1.0-Diffusion-7B-Text2World"
        pipe = CosmosTextToWorldPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        # pipe.to("cuda")
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_tiling()
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe._guidance_scale = 1.0 if not use_cfg else 7.0
    else:
        raise ValueError(f"Invalid model: {model_type}")

    # override the guidance strength if provided manually
    if args.guidance_strength != -1:
        pipe._guidance_scale = args.guidance_strength

    return pipe


        
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="svd")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--data", type=str, default="ball_drop", help="Data configuration to use")
    parser.add_argument("--guidance_scale", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--num_frames", type=int, default=-1, help="Number of frames to evaluate")
    parser.add_argument("--height", type=int, default=-1, help="Height of the video")
    parser.add_argument("--width", type=int, default=-1, help="Width of the video")
    parser.add_argument("--size_scale", type=int, default=1, help="Size scale of the video")
    parser.add_argument("--fps", type=int, default=-1, help="FPS of the video")
    parser.add_argument("--timestep_strategy", type=str, default="uniform", help="Timestep sampling strategy of the video")
    parser.add_argument("--timestep_num", type=int, default=10, help="Number of timesteps to evaluate")
    parser.add_argument("--guidance_strength", type=int, default=-1, help="The strength of the guidance")
    parser.add_argument("--tag_name", type=str, default=" ", help="Name of the ablation study")
    parser.add_argument("--exp_name", type=str, default="evaluation_t10_uniform", help="Name of the experiment")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")

    parser.add_argument("--prompt_exp", type=str, default="no", help="for prompt exp")
    
    return parser.parse_args()

if __name__ == "__main__":
    import sys
    import os
    import json

    args = parse_args()
    
    # Get model-specific parameters
    model_params = get_model_params(args.model)

    if args.width == -1:
        args.width = model_params["width"]
        
    if args.height == -1:
        args.height = model_params["height"]
    if args.fps == -1:
        args.fps = model_params["fps"]
    if args.num_frames == -1:
        args.num_frames = model_params["num_frames"]

    args.width = args.width * args.size_scale
    args.height = args.height * args.size_scale

    # Data configuration for different datasets
    data_config = {
        "ball_drop": {
            "dataset_dir": "./data/ball_drop_videos",
            "data_name": "ball_drop"
        },
        "ball_collision": {
            "dataset_dir": "./data/ball_collision_videos",
            "data_name": "ball_collision"
        },
        "pendulum": {
            "dataset_dir": "./data/pendulum_videos",
            "data_name": "pendulum"
        },
        "block_slide": {
            "dataset_dir": "./data/block_slide_videos",
            "data_name": "block_slide"
        },
        "fluid": {
            "dataset_dir": "./data/fluid_videos",
            "data_name": "fluid"
        },
        "faucet": {
            "dataset_dir": "./data/faucet_videos",
            "data_name": "faucet"
        },
        "cloth": {
            "dataset_dir": "./data/cloth_drape_videos",
            "data_name": "cloth"
        },
        "flag": {
            "dataset_dir": "./data/flag_videos",
            "data_name": "flag"
        },
        "river": {
            "dataset_dir": "./data/river_videos",
            "data_name": "river"
        },
        "shadow": {
            "dataset_dir": "./data/shadow_videos",
            "data_name": "shadow"
        },
        "pyramid": {
            "dataset_dir": "./data/pyramid_videos",
            "data_name": "pyramid"
        },
        "shadowm": {
            "dataset_dir": "./data/shadow_camera_videos",
            "data_name": "shadowm"
        },
        'sample':{
            'dataset_dir': "./data/abluse",
            'data_name':'abluse'
        }
    }

    # Validate data choice
    if args.data not in data_config:
        raise ValueError(f"Invalid data configuration: {args.data}. Available options: {list(data_config.keys())}")
    
    current_data_config = data_config[args.data]
    dataset_dir = current_data_config["dataset_dir"]
    data_name = current_data_config["data_name"]

    # Experiment naming
    cfg_tag = "cfg" if args.guidance_scale else "no_cfg"
    if args.tag_name != " ":
        exp_name = f"{args.exp_name}_{args.seed}_{cfg_tag}_{args.tag_name}"
    else:
        exp_name = f"{args.exp_name}_{args.seed}_{cfg_tag}"

    # Determine where to save results
    output_file = f"./{args.output_dir}/{exp_name}/{data_name}/results_{args.model}.json"

    # Skip if results already exist and appear complete
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                saved = json.load(f)
            if "scene_evaluations" in saved and "misrank_metrics" in saved:
                print(f"Results already exist and look complete at {output_file}. Skipping evaluation.")
                sys.exit(0)
        except Exception as e:
            print(f"Found existing file at {output_file} but couldn't validate it ({e}), continuing with evaluation...")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Optional Weights & Biases logging
    USE_WANDB = args.use_wandb
    if USE_WANDB:
        import wandb
        run_name = f"{args.model}_evaluation"
        wandb.init(
            project="physical-eval", 
            name=run_name,
            config={
                "model": args.model,
                "height": args.height,
                "width": args.width,
                "fps": args.fps,
                "num_frames": args.num_frames,
                "data": args.data
            }
        )

    # Initialize model pipeline
    pipe = initialize_model(args)
    
    # Set global seed
    _ = set_seed(args.seed)

    # Run evaluation
    results = evaluate_dataset(args, dataset_dir, pipe)
    misrank_metrics = compute_misrank_normalized(results)
    
    # Combine and save
    final_results = {
        "scene_evaluations": results,
        "misrank_metrics": misrank_metrics
    }
    with open(output_file, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Finalize wandb
    if USE_WANDB:
        wandb.config.update({"model": args.model, "data": args.data})
        wandb.finish()
