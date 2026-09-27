"""
Arbitrary-aspect-ratio inference wrapper around StereoSpace
(Behrens et al., "StereoSpace: Depth-Free Synthesis of Stereo Geometry via
End-to-End Diffusion in a Canonical Space", arXiv:2512.10959, MIT license).

Differences from the reference `src/StereoSpace.py`:
  * loads from a local directory (no forced HF repo id at inference time)
  * latents are allocated at H/8 x W/8 instead of a hard-coded 768x768 square,
    and explicit pixel intrinsics (fx == fy) are always passed so that the
    Plücker rays stay physically correct for non-square frames
  * deterministic seeding, per-step callback, optional CPU offload
"""
import os
from typing import Callable, Optional

import torch
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from .geometry import get_plucker_coordinates, normalize_K
from .models import ReferenceAttentionControl, UNet2DConditionModel, UNet3DConditionModel

HF_REPO = "prs-eth/stereospace-v1-0"
CLIP_SUBFOLDER = "CLIP-ViT-H-14-laion2B-s32B-b79K"
HF_ALLOW_PATTERNS = [
    "config.json",
    "reference_unet.pth",
    "denoising_unet.pth",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    f"{CLIP_SUBFOLDER}/config.json",
    f"{CLIP_SUBFOLDER}/preprocessor_config.json",
    f"{CLIP_SUBFOLDER}/model.safetensors",
]

SCHEDULER_KWARGS = dict(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    steps_offset=1,
    clip_sample=False,
    # enable_zero_snr: True in the reference config
    rescale_betas_zero_snr=True,
    timestep_spacing="trailing",
    prediction_type="v_prediction",
)


def ensure_weights(model_dir: str) -> str:
    needed = ["config.json", "reference_unet.pth", "denoising_unet.pth"]
    if all(os.path.isfile(os.path.join(model_dir, f)) for f in needed):
        return model_dir
    from huggingface_hub import snapshot_download

    os.makedirs(model_dir, exist_ok=True)
    print(f"[StereoForge] downloading {HF_REPO} (~11 GB) to {model_dir}")
    snapshot_download(HF_REPO, local_dir=model_dir, allow_patterns=HF_ALLOW_PATTERNS)
    return model_dir


class StereoSpacePipeline:
    def __init__(self, model_dir: str, device: torch.device, dtype: torch.dtype):
        self.model_dir = model_dir
        self.device = device
        self.dtype = dtype
        self._load()

    def _load(self):
        d, dt = self.model_dir, self.dtype
        self.vae = AutoencoderKL.from_pretrained(d, subfolder="vae", torch_dtype=dt)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            d, subfolder=CLIP_SUBFOLDER, torch_dtype=dt
        )
        self.clip_image_processor = CLIPImageProcessor()

        config = UNet2DConditionModel.load_config(d)
        self.reference_unet = UNet2DConditionModel.from_config(config)
        self.reference_unet.load_state_dict(
            torch.load(os.path.join(d, "reference_unet.pth"), map_location="cpu", weights_only=True)
        )
        self.reference_unet.to(dtype=dt)

        cfg3d = dict(config)
        cfg3d["_class_name"] = UNet3DConditionModel.__name__
        cfg3d["down_block_types"] = ["CrossAttnDownBlock3D"] * 3 + ["DownBlock3D"]
        cfg3d["up_block_types"] = ["UpBlock3D"] + ["CrossAttnUpBlock3D"] * 3
        cfg3d["mid_block_type"] = "UNetMidBlock3DCrossAttn"
        self.denoising_unet = UNet3DConditionModel.from_config(
            cfg3d,
            use_motion_module=False,
            unet_use_temporal_attention=False,
            use_zero_convs=False,
        )
        sd = torch.load(os.path.join(d, "denoising_unet.pth"), map_location="cpu", weights_only=True)
        missing, unexpected = self.denoising_unet.load_state_dict(sd, strict=False)
        if unexpected:
            print(f"[StereoForge] warning: {len(unexpected)} unexpected keys in denoising_unet")
        self.denoising_unet.to(dtype=dt)
        del sd

        for m in self.modules():
            m.requires_grad_(False)
            m.eval()
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def modules(self):
        return [self.vae, self.image_encoder, self.reference_unet, self.denoising_unet]

    def to(self, device):
        for m in self.modules():
            m.to(device)
        self.device = torch.device(device)
        return self

    # ------------------------------------------------------------------ #
    def _plucker(self, baseline: float, H: int, W: int, K_px: torch.Tensor):
        c2w = torch.eye(4, device=self.device).unsqueeze(0).repeat(2, 1, 1)
        c2w[0, 0, 3] = -0.5 * baseline
        c2w[1, 0, 3] = 0.5 * baseline
        w2c = torch.linalg.inv(c2w)
        Kn = normalize_K(K_px.unsqueeze(0).to(self.device), H, W)
        Ks = torch.cat([Kn, Kn], 0)
        F = self.vae_scale_factor
        pl = get_plucker_coordinates(
            extrinsics_src=w2c[0], extrinsics=w2c, intrinsics=Ks, target_size=(H // F, W // F)
        )
        return pl[0:1], pl[1:2]

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,  # [1,3,H,W] in [0,1], H and W multiples of 64
        baseline: float,  # metres, +ve -> target camera to the right of source
        focal_px: float,
        steps: int = 50,
        guidance: float = 1.5,
        seed: int = 0,
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> torch.Tensor:
        _, _, H, W = src.shape
        assert H % 64 == 0 and W % 64 == 0, "StereoSpace input must be a multiple of 64"
        dev, dt = self.device, self.dtype
        src_n = (src.to(dev, dt) * 2 - 1)

        clip_in = self.clip_image_processor(
            images=src.float().cpu().clamp(0, 1), return_tensors="pt", do_rescale=False
        ).pixel_values.to(dev, dt)
        emb = self.image_encoder(clip_in).image_embeds.unsqueeze(1)
        prompt = torch.cat([torch.zeros_like(emb), emb], 0)

        sched = DDIMScheduler(**SCHEDULER_KWARGS)
        sched.set_timesteps(steps, device=dev)

        g = torch.Generator(device="cpu").manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)
        F = self.vae_scale_factor
        noise = torch.randn((1, 4, H // F, W // F), generator=g).to(dev, dt)
        latents = noise  # pure noise start (reference adds noise to random latents at t=T; equivalent under zero-SNR)

        ref_lat = self.vae.encode(src_n).latent_dist.mean * 0.18215

        K = torch.tensor(
            [[focal_px, 0, W / 2.0], [0, focal_px, H / 2.0], [0, 0, 1.0]], dtype=torch.float32
        )
        src_pl, tgt_pl = self._plucker(float(baseline), H, W, K)
        src_pl, tgt_pl = src_pl.to(dev, dt), tgt_pl.to(dev, dt)

        reader = ReferenceAttentionControl(
            self.denoising_unet, do_classifier_free_guidance=True, mode="read",
            batch_size=1, fusion_blocks="full", feature_fusion_type="attention_full_sharing",
        )
        writer = ReferenceAttentionControl(
            self.reference_unet, do_classifier_free_guidance=True, mode="write",
            batch_size=1, fusion_blocks="full", feature_fusion_type="attention_full_sharing",
        )
        try:
            ref_in = torch.cat([ref_lat, src_pl], 1)
            self.reference_unet(
                ref_in.repeat(2, 1, 1, 1),
                torch.zeros(2, device=dev, dtype=dt),
                encoder_hidden_states=prompt,
                dense_emb=src_pl.repeat(2, 1, 1, 1),
                return_dict=False,
            )
            reader.update(writer)
            n = len(sched.timesteps)
            for i, t in enumerate(sched.timesteps):
                x = sched.scale_model_input(latents, t)
                x = torch.cat([x, tgt_pl], 1).unsqueeze(2)
                x = torch.cat([x, x])
                pred = self.denoising_unet(
                    x, t, encoder_hidden_states=prompt,
                    dense_emb=tgt_pl.repeat(2, 1, 1, 1), return_dict=False,
                )[0].squeeze(2)
                u, c = pred.chunk(2)
                pred = u + guidance * (c - u)
                latents = sched.step(pred, t, latents, return_dict=False)[0]
                if callback is not None:
                    callback(i + 1, n)
        finally:
            reader.clear()
            writer.clear()

        img = self.vae.decode(latents / 0.18215).sample
        return (img.float() / 2 + 0.5).clamp(0, 1)
