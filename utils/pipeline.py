import inspect, math
from typing import Callable, List, Optional, Union
from dataclasses import dataclass
from PIL import Image
import numpy as np
import torch
from einops import rearrange
import torch.distributed as dist
from tqdm import tqdm
from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput

from ref_encoder.latent_controlnet import ControlNetModel
from ref_encoder.reference_control import ReferenceAttentionControl
import torch.nn.functional as F

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class PipelineOutput(BaseOutput):
    samples: Union[torch.Tensor, np.ndarray]


class StableHairPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: UNet2DConditionModel,
            controlnet: ControlNetModel,
            scheduler: Union[
                DDIMScheduler,
                PNDMScheduler,
                LMSDiscreteScheduler,
                EulerDiscreteScheduler,
                EulerAncestralDiscreteScheduler,
                DPMSolverMultistepScheduler,
            ],
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                    hasattr(module, "_hf_hook")
                    and hasattr(module._hf_hook, "execution_device")
                    and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, prompt, device, do_classifier_free_guidance, negative_prompt):
        if isinstance(prompt, torch.Tensor):
            batch_size = prompt.shape[0]
            text_input_ids = prompt
        else:
            batch_size = 1 if isinstance(prompt, str) else len(prompt)
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids,
                                                                                         untruncated_ids):
                removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1: -1])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1)
        image = image.cpu().squeeze(0).float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list) and not isinstance(prompt, torch.Tensor):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
                callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                self.vae.encode(image[i: i + 1]).latent_dist.sample(generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = self.vae.encode(image).latent_dist.sample(generator=generator)

        image_latents = self.vae.config.scaling_factor * image_latents

        return image_latents

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None,
                        clip_length=16):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)

        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        noise = latents.clone()
        latents = latents * self.scheduler.init_noise_sigma
        return latents, noise

    def prepare_condition(self, condition, device, dtype, do_classifier_free_guidance):
        if isinstance(condition, torch.Tensor):
            # suppose input is [-1, 1]
            condition = condition
        elif isinstance(condition, np.ndarray):
            # suppose input is [0, 255]
            condition = self.images2latents(condition, dtype).cuda()
        if do_classifier_free_guidance:
            condition_pad = torch.ones_like(condition) * -1
            condition = torch.cat([condition_pad, condition])
        return condition

    @torch.no_grad()
    def images2latents(self, images, dtype):
        """
        Convert RGB image to VAE latents
        """
        device = self._execution_device
        if isinstance(images, torch.Tensor):
            # suppose input is [-1, 1]
            images = images.to(dtype)
            if images.ndim == 3:
                images = images.unsqueeze(0)
        elif isinstance(images, np.ndarray):
            # suppose input is [0, 255]
            images = torch.from_numpy(images).float().to(dtype) / 127.5 - 1
            images = rearrange(images, "h w c -> c h w").to(device)[None, :]
        latents = self.vae.encode(images)['latent_dist'].mean * self.vae.config.scaling_factor
        return latents

    @torch.no_grad()
    def encode_single_image_latents(self, images, mask, dtype):
        device = self._execution_device
        images = torch.from_numpy(images).float().to(dtype) / 127.5 - 1
        images = rearrange(images, "h w c -> c h w").to(device)
        latents = self.vae.encode(images[None, :])['latent_dist'].mean * 0.18215

        images = images.unsqueeze(0)

        mask = torch.from_numpy(mask).float().to(dtype).to(device) / 255.0
        if mask.ndim == 2:
            mask = mask[None, None, :]
        elif mask.ndim == 3:
            mask = mask[:, None, :, :]

        mask = F.interpolate(mask, size=latents.shape[-2:], mode='nearest')
        return latents, images, mask

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]],
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "np",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            controlnet_condition: list = None,
            controlnet_conditioning_scale: Optional[float] = 1.0,
            init_latents: Optional[torch.FloatTensor] = None,
            num_actual_inference_steps: Optional[int] = None,
            reference_encoder=None,
            ref_image=None,
            t2i=False,
            style_fidelity=1.0,
            **kwargs,
    ):
        controlnet = self.controlnet

        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # Define call parameters
        # batch_size = 1 if isinstance(prompt, str) else len(prompt)
        batch_size = 1
        if latents is not None:
            batch_size = latents.shape[0]
        if isinstance(prompt, list):
            batch_size = len(prompt)

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input prompt
        if not isinstance(prompt, torch.Tensor):
            prompt = prompt if isinstance(prompt, list) else [prompt] * batch_size
        if negative_prompt is not None:
            negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * batch_size
        text_embeddings = self._encode_prompt(
            prompt, device, do_classifier_free_guidance, negative_prompt
        )
        text_embeddings = torch.cat([text_embeddings])

        reference_control_writer = ReferenceAttentionControl(reference_encoder, do_classifier_free_guidance=True,
                                                             style_fidelity=style_fidelity,
                                                             mode='write', fusion_blocks='full')
        reference_control_reader = ReferenceAttentionControl(self.unet, do_classifier_free_guidance=True, mode='read',
                                                             style_fidelity=style_fidelity,
                                                             fusion_blocks='full')

        is_dist_initialized = kwargs.get("dist", False)
        rank = kwargs.get("rank", 0)

        # Prepare control_img
        control = self.prepare_condition(
            condition=controlnet_condition,
            device=device,
            dtype=controlnet.dtype,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )
        # for b in range(control.size(0)):
        #     max_value = torch.max(control[b])
        #     min_value = torch.min(control[b])
        #     control[b] = (control[b] - min_value) / (max_value - min_value)

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )
        if isinstance(latents, tuple):
            latents, noise = latents

        latents_dtype = latents.dtype

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # For img2img setting
        if num_actual_inference_steps is None:
            num_actual_inference_steps = num_inference_steps

        if isinstance(ref_image, str):
            ref_image_latents = self.images2latents(np.array(Image.open(ref_image).resize((width, height))),
                                                    latents_dtype).cuda()
        elif isinstance(ref_image, np.ndarray):
            ref_image_latents = self.images2latents(ref_image, latents_dtype).cuda()
        elif isinstance(ref_image, torch.Tensor):
            ref_image_latents = self.images2latents(ref_image, latents_dtype).cuda()

        ref_padding_latents = torch.ones_like(ref_image_latents) * -1
        ref_image_latents = torch.cat([ref_padding_latents, ref_image_latents]) if do_classifier_free_guidance else ref_image_latents

        # Denoising loop
        for i, t in tqdm(enumerate(timesteps), total=len(timesteps), disable=(rank != 0)):
            if num_actual_inference_steps is not None and i < num_inference_steps - num_actual_inference_steps:
                continue

            # writer
            ref_latents_input = ref_image_latents
            reference_encoder(
                ref_latents_input,
                t,
                encoder_hidden_states=text_embeddings,
                return_dict=False,
            )
            reference_control_reader.update(reference_control_writer)

            # prepare latents
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

            if t2i:
                pass

            else:
                # controlnet
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=control,
                    return_dict=False,
                )
                down_block_res_samples = [sample * controlnet_conditioning_scale for sample in down_block_res_samples]
                mid_block_res_sample = mid_block_res_sample * controlnet_conditioning_scale

            if t2i:
                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    return_dict=False,
                )[0]

            else:
                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]

            # clean the reader
            reference_control_reader.clear()

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            if is_dist_initialized:
                dist.broadcast(latents, 0)
                dist.barrier()

            reference_control_writer.clear()

        samples = self.decode_latents(latents)
        if is_dist_initialized:
            dist.barrier()

        # Convert to tensor
        if output_type == "tensor":
            samples = torch.from_numpy(samples)

        if not return_dict:
            return samples

        return PipelineOutput(samples=samples)
