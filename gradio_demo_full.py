import gradio as gr
import torch
from PIL import Image
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import os
import cv2
from diffusers import DDIMScheduler, UniPCMultistepScheduler
from diffusers.models import UNet2DConditionModel
from ref_encoder.latent_controlnet import ControlNetModel
from ref_encoder.adapter import *
from ref_encoder.reference_unet import ref_unet
from utils.pipeline import StableHairPipeline
from utils.pipeline_cn import StableDiffusionControlNetPipeline


class StableHair:
    def __init__(self, config="./configs/hair_transfer.yaml", device="cuda", weight_dtype=torch.float32) -> None:
        print("Initializing Stable Hair Pipeline...")
        self.config = OmegaConf.load(config)
        self.device = device

        ### Load vae controlnet
        unet = UNet2DConditionModel.from_pretrained(self.config.pretrained_model_path, subfolder="unet").to(device)
        controlnet = ControlNetModel.from_unet(unet).to(device)
        _state_dict = torch.load(os.path.join(self.config.pretrained_folder, self.config.controlnet_path))
        controlnet.load_state_dict(_state_dict, strict=False)
        controlnet.to(weight_dtype)

        ### >>> create pipeline >>> ###
        self.pipeline = StableHairPipeline.from_pretrained(
            self.config.pretrained_model_path,
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=weight_dtype,
        ).to(device)
        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)

        ### load Hair encoder/adapter
        self.hair_encoder = ref_unet.from_pretrained(self.config.pretrained_model_path, subfolder="unet").to(device)
        _state_dict = torch.load(os.path.join(self.config.pretrained_folder, self.config.encoder_path))
        self.hair_encoder.load_state_dict(_state_dict, strict=False)
        self.hair_adapter = adapter_injection(self.pipeline.unet, device=self.device, dtype=torch.float16, use_resampler=False)
        _state_dict = torch.load(os.path.join(self.config.pretrained_folder, self.config.adapter_path))
        self.hair_adapter.load_state_dict(_state_dict, strict=False)

        ### load bald converter
        bald_converter = ControlNetModel.from_unet(unet).to(device)
        _state_dict = torch.load(self.config.bald_converter_path)
        bald_converter.load_state_dict(_state_dict, strict=False)
        bald_converter.to(dtype=weight_dtype)
        del unet

        ### create pipeline for hair removal
        self.remove_hair_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.config.pretrained_model_path,
            controlnet=bald_converter,
            safety_checker=None,
            torch_dtype=weight_dtype,
        )
        self.remove_hair_pipeline.scheduler = UniPCMultistepScheduler.from_config(self.remove_hair_pipeline.scheduler.config)
        self.remove_hair_pipeline = self.remove_hair_pipeline.to(device)

        ### move to fp16
        self.hair_encoder.to(weight_dtype)
        self.hair_adapter.to(weight_dtype)

        print("Initialization Done!")

    def Hair_Transfer(self, source_image, reference_image, random_seed, step, guidance_scale, scale, controlnet_conditioning_scale):
        prompt = ""
        n_prompt = ""
        random_seed = int(random_seed)
        step = int(step)
        guidance_scale = float(guidance_scale)
        scale = float(scale)
        controlnet_conditioning_scale = float(controlnet_conditioning_scale)

        # load imgs
        H, W, C = source_image.shape

        # generate images
        set_scale(self.pipeline.unet, scale)
        generator = torch.Generator(device="cuda")
        generator.manual_seed(random_seed)
        sample = self.pipeline(
            prompt,
            negative_prompt=n_prompt,
            num_inference_steps=step,
            guidance_scale=guidance_scale,
            width=W,
            height=H,
            controlnet_condition=source_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
            reference_encoder=self.hair_encoder,
            ref_image=reference_image,
        ).samples
        return sample, source_image, reference_image

    def get_bald(self, id_image, scale):
        H, W = id_image.size
        scale = float(scale)
        image = self.remove_hair_pipeline(
            prompt="",
            negative_prompt="",
            num_inference_steps=30,
            guidance_scale=1.5,
            width=W,
            height=H,
            image=id_image,
            controlnet_conditioning_scale=scale,
            generator=None,
        ).images[0]

        return image


model = StableHair(config="./configs/hair_transfer.yaml", weight_dtype=torch.float32)

# Define your ML model or function here
def model_call(id_image, ref_hair, converter_scale, scale, guidance_scale, controlnet_conditioning_scale):
    # # Your ML logic goes here
    id_image = Image.fromarray(id_image.astype('uint8'), 'RGB')
    ref_hair = Image.fromarray(ref_hair.astype('uint8'), 'RGB')
    id_image = id_image.resize((512, 512))
    ref_hair = ref_hair.resize((512, 512))
    id_image_bald = model.get_bald(id_image, converter_scale)

    id_image_bald = np.array(id_image_bald)
    ref_hair = np.array(ref_hair)

    image, source_image, reference_image = model.Hair_Transfer(source_image=id_image_bald,
                                                               reference_image=ref_hair,
                                                               random_seed=-1,
                                                               step=30,
                                                               guidance_scale=guidance_scale,
                                                               scale=scale,
                                                               controlnet_conditioning_scale=controlnet_conditioning_scale
                                                               )

    image = Image.fromarray((image * 255.).astype(np.uint8))
    return id_image_bald, image

# Create a Gradio interface
image1 = gr.inputs.Image(label="id_image")
image2 = gr.inputs.Image(label="ref_hair")
number0 = gr.inputs.Slider(minimum=0.5, maximum=1.5, default=1, label="Converter Scale")
number1 = gr.inputs.Slider(minimum=0.0, maximum=3, default=1.0, label="Hair Encoder Scale")
number2 = gr.inputs.Slider(minimum=1.1, maximum=3.0, default=1.5, label="CFG")
number3 = gr.inputs.Slider(minimum=0.1, maximum=2.0, default=1, label="Latent IdentityNet Scale")
output1 = gr.outputs.Image(type="pil", label="Bald_Result")
output2 = gr.outputs.Image(type="pil", label="Transfer Result")

iface = gr.Interface(
    fn=lambda id_image, ref_hair, num0, num1, num2, num3, : model_call(id_image, ref_hair, num0, num1, num2, num3),
    inputs=[image1, image2, number0, number1, number2, number3],
    outputs=[output1, output2],
    title="Hair Transfer Demo",
    description="In general, aligned faces work well, but can also be used on non-aligned faces, and you need to resize to 512 * 512"
)

# Launch the Gradio interface
iface.queue().launch(server_name='0.0.0.0', server_port=8986)
