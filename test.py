import argparse
#sv3d_p import
import math
import os
import sys
from glob import glob
from pathlib import Path
from typing import List, Optional

import cv2
import imageio
import numpy as np
import torch
from einops import rearrange, repeat
from fire import Fire
from omegaconf import OmegaConf
from PIL import Image
from rembg import remove
from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from sgm.inference.helpers import embed_watermark
from sgm.util import default, instantiate_from_config
from torchvision.transforms import ToTensor
#LGM import
import tyro
import tqdm
import torch.nn as nn
import torch.nn.functional as Func
import torchvision.transforms.functional as TF
from safetensors.torch import load_file
import gradio as gr

import kiui
from kiui.op import recenter
from kiui.cam import orbit_camera

from core.options import AllConfigs, Options,config_defaults
from core.models import LGM
from mvdream.pipeline_mvdream import MVDreamPipeline
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames" or key == "cond_frames_without_noise":
            batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=N[0])
        elif key == "polars_rad" or key == "azimuths_rad":
            batch[key] = torch.tensor(value_dict[key]).to(device).repeat(N[0])
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def load_model(
    config: str,
    device: str,
    num_frames: int,
    num_steps: int,
    verbose: bool = False,
):
    config = OmegaConf.load(config)
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.verbose = verbose
    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )
    if device == "cuda":
        with torch.device(device):
            model = instantiate_from_config(config.model).to(device).eval()
    else:
        model = instantiate_from_config(config.model).to(device).eval()

    filter = DeepFloydDataFiltering(verbose=False, device=device)
    return model, filter

def test(args):
    elevations_deg=[10]*args.num_frames
    polars_rad=[np.deg2rad(90-e) for e in elevations_deg]
    azimuths_deg=np.linspace(0,360,args.num_frames+1)[1:]%360
    azimuths_rad=[np.deg2rad((a-azimuths_deg[-1])%360) for a in azimuths_deg]

    sv3d_model,filter=load_model(
        args.model_config,
        args.device,
        args.num_frames,
        args.num_steps,
        verbose=False,
    )
    torch.manual_seed(args.seed)

    assert args.input_path is not None
    assert os.path.exists(os.path.join(args.input_path))

    if os.path.exists(os.path.join(args.input_path)):
        image=Image.open(args.input_path)
    else:
        raise FileNotFoundError(f"Could not find file {args.input_path}")

    if image.mode == "RGBA":
        pass
    else:
        # remove bg
        image.thumbnail([768, 768], Image.Resampling.LANCZOS)
        image = remove(image.convert("RGBA"), alpha_matting=True)

    # resize object in frame
    image_arr = np.array(image)
    in_w, in_h = image_arr.shape[:2]
    ret, mask = cv2.threshold(
        np.array(image.split()[-1]), 0, 255, cv2.THRESH_BINARY
    )
    x, y, w, h = cv2.boundingRect(mask)
    max_size = max(w, h)
    side_len = (
        int(max_size / args.image_frame_ratio)
        if args.image_frame_ratio is not None
        else in_w
    )
    padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
    center = side_len // 2
    padded_image[
        center - h // 2 : center - h // 2 + h,
        center - w // 2 : center - w // 2 + w,
    ] = image_arr[y : y + h, x : x + w]
    # resize frame to 576x576
    rgba = Image.fromarray(padded_image).resize((576, 576), Image.LANCZOS)
    # white bg
    rgba_arr = np.array(rgba) / 255.0
    rgb = rgba_arr[..., :3] * rgba_arr[..., -1:] + (1 - rgba_arr[..., -1:])
    input_image = Image.fromarray((rgb * 255).astype(np.uint8))
    
    image = ToTensor()(input_image)
    image = image * 2.0 - 1.0

    image = image.unsqueeze(0).to(args.device)
    H, W = image.shape[2:]
    assert image.shape[1] == 3
    F = 8
    C = 4
    shape = (args.num_frames, C, H // F, W // F)

    value_dict = {}
    value_dict["cond_frames_without_noise"] = image
    value_dict["motion_bucket_id"] = args.motion_bucket_id
    value_dict["fps_id"] = args.fps_id
    value_dict["cond_aug"] = args.cond_aug
    value_dict["cond_frames"] = image + args.cond_aug * torch.randn_like(image)
    value_dict["polars_rad"] = polars_rad
    value_dict["azimuths_rad"] = azimuths_rad   
    
    with torch.no_grad():
        with torch.autocast(args.device):
            batch, batch_uc = get_batch(
                get_unique_embedder_keys_from_conditioner(sv3d_model.conditioner),
                value_dict,
                [1, args.num_frames],
                T=args.num_frames,
                device=args.device,
            )
            c, uc = sv3d_model.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=[
                    "cond_frames",
                    "cond_frames_without_noise",
                ],
            )

            for k in ["crossattn", "concat"]:
                uc[k] = repeat(uc[k], "b ... -> b t ...", t=args.num_frames)
                uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=args.num_frames)
                c[k] = repeat(c[k], "b ... -> b t ...", t=args.num_frames)
                c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=args.num_frames)

            randn = torch.randn(shape, device=args.device)

            additional_model_inputs = {}
            additional_model_inputs["image_only_indicator"] = torch.zeros(
                2, args.num_frames
            ).to(args.device)
            additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

            def denoiser(input, sigma, c):
                return sv3d_model.denoiser(
                    sv3d_model.model, input, sigma, c, **additional_model_inputs
                )

            samples_z = sv3d_model.sampler(denoiser, randn, cond=c, uc=uc)
            sv3d_model.en_and_decode_n_samples_a_time = args.decoding_t
            samples_x = sv3d_model.decode_first_stage(samples_z)

            samples_x[-1:] = value_dict["cond_frames_without_noise"]
            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

            samples = embed_watermark(samples)
            samples = filter(samples) # t c h w [21,3,576,576]

    sv3d_model= sv3d_model.cpu()
    torch.cuda.empty_cache()
        
    os.makedirs(args.output_path,exist_ok=True)
    output_ply_path=os.path.join(args.output_path,"output.ply")
    output_video_path=os.path.join(args.output_path,"output.mp4")

    opt = config_defaults['big']
    opt.resume=args.lgm_checkpoints_path
    opt.num_frames=args.num_frames

    # model
    lgm_model = LGM(opt)
    # resume pretrained checkpoint
    if opt.resume is not None:
        if opt.resume.endswith('safetensors'):
            ckpt = load_file(opt.resume, device='cpu')
        else:
            ckpt = torch.load(opt.resume, map_location='cpu')
        lgm_model.load_state_dict(ckpt, strict=False)
        print(f'[INFO] Loaded checkpoint from {opt.resume}')
    else:
        print(f'[WARN] model randomly initialized, are you sure?')

    lgm_model = lgm_model.half().to(args.device)
    lgm_model.eval()

    tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
    proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=args.device)
    proj_matrix[0, 0] = 1 / tan_half_fov
    proj_matrix[1, 1] = 1 / tan_half_fov
    proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[2, 3] = 1

    input_image = Func.interpolate(samples, size=(opt.input_size, opt.input_size), mode='bilinear', align_corners=False)
    input_image = TF.normalize(input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    rays_embeddings = lgm_model.prepare_default_rays(args.device,num_frames=opt.num_frames,elevation=10)
    input_image = torch.cat([input_image, rays_embeddings], dim=1).unsqueeze(0) # [1, 4, 9, H, W]

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # generate gaussians
            gaussians = lgm_model.forward_gaussians(input_image)

        # save gaussians
        lgm_model.gs.save_ply(gaussians, output_ply_path)

    # render 360 video 
    images = []
    elevation = 0
    if opt.fancy_video:
        azimuth = np.arange(0, 720, 4, dtype=np.int32)
        for azi in tqdm.tqdm(azimuth):

            cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(args.device)

            cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction

            # cameras needed by gaussian rasterizer
            cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
            cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
            cam_pos = - cam_poses[:, :3, 3] # [V, 3]

            scale = min(azi / 360, 1)

            image = lgm_model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=scale)['image']
            images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))
    else:
        azimuth = np.arange(0, 360, 2, dtype=np.int32)
        for azi in tqdm.tqdm(azimuth):

            cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(args.device)

            cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction

            # cameras needed by gaussian rasterizer
            cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
            cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
            cam_pos = - cam_poses[:, :3, 3] # [V, 3]

            image = lgm_model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)['image']
            images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))

    images = np.concatenate(images, axis=0)
    r=images[:, :, :, 0]
    g=images[:, :, :, 1]
    b=images[:, :, :, 2]
    images_cv=np.stack([b,g,r],axis=-1)
    N_FPS=images_cv.shape[0]
    media_writer=cv2.VideoWriter(output_video_path,cv2.VideoWriter_fourcc(*'mp4v'),30,(images_cv.shape[2],images_cv.shape[1]))
    for i in range(N_FPS):
        media_writer.write(images_cv[i])
    media_writer.release()
    pass

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--input_path',type=str,default='test_model_images/239.png')
    parser.add_argument('--num_frames',type=int,default=21)
    parser.add_argument('--model_config',type=str,default='configs/sv3d_p.yaml')
    parser.add_argument('--num_steps',type=int,default=50)
    parser.add_argument('--device',type=str,default='cuda')
    parser.add_argument('--elevation',type=int,default=10)
    parser.add_argument('--output_path',type=str,default='outputs')
    parser.add_argument('--image_frame_ratio',type=int,default=None)
    parser.add_argument('--motion_bucket_id',type=int,default=127)
    parser.add_argument('--fps_id',type=int,default=6)
    parser.add_argument('--cond_aug',type=float,default=0.02)
    parser.add_argument('--decoding_t',type=int,default=6)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--lgm_checkpoints_path',type=str,default='pretrained/model_fp16.safetensors')    
    args=parser.parse_args()
    test(args)
