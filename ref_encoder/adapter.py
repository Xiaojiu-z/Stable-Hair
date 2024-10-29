import torch
import torch.nn.functional as F
def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")
if is_torch2_available():
    from .attention_processor import HairAttnProcessor2_0 as HairAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from .attention_processor import HairAttnProcessor, AttnProcessor

def adapter_injection(unet, device="cuda", dtype=torch.float32, use_resampler=False):
    device = device
    dtype = dtype
    # load Hair attention layers
    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = HairAttnProcessor(hidden_size=hidden_size, cross_attention_dim=hidden_size, scale=1, use_resampler=use_resampler).to(device, dtype=dtype)
        else:
            attn_procs[name] = AttnProcessor()
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    adapter_layers = adapter_modules
    adapter_layers.to(device, dtype=dtype)
    return adapter_layers

def set_scale(unet, scale):
    for attn_processor in unet.attn_processors.values():
        if isinstance(attn_processor, HairAttnProcessor):
            attn_processor.scale = scale