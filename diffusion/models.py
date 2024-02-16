#import diffusers
from diffusers import StableDiffusionPipeline,EulerDiscreteScheduler
import torch

MODEL_IDS = {
    '2-1': "stabilityai/stable-diffusion-2-1-base"
}


def get_sd_model():
    
    dtype = torch.float32
    

    
    model_id = MODEL_IDS['2-1']
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=dtype)
    pipe.enable_xformers_memory_efficient_attention()
    vae = pipe.vae
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    unet = pipe.unet
    return vae, tokenizer, text_encoder, unet, scheduler


def get_scheduler_config():
   
    config = {
        "_class_name": "EulerDiscreteScheduler",
        "_diffusers_version": "0.14.0",
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "beta_start": 0.00085,
        "interpolation_type": "linear",
        "num_train_timesteps": 1000,
        "prediction_type": "epsilon",
        "set_alpha_to_one": False,
        "skip_prk_steps": True,
        "steps_offset": 1,
        "trained_betas": None
        }
    

    return config
