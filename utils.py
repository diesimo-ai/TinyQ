import os
import logging
from platform import processor
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import pipeline, \
      AutoTokenizer, AutoModelForCausalLM, \
      AutoFeatureExtractor, AutoForAudioClassification 

import requests
import argparse

from PIL import Image

import warnings

#-------------------------------------------------------#
#           Preprocessing utilities                     #
#-------------------------------------------------------#

def setup_logging(name: str, log_dir: str = "logs"):
    """
    Setup logging configuration
    Args:
        name: Logger name (e.g. 'tinyq', 'benchmark', 'quantizer')
        log_dir: Base directory for logs
    """
    log_path = os.path.join(log_dir, f"{name}.log")
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name)

def load_image(img_url):
    image = Image.open(requests.get(url=img_url, stream=True).raw).convert("RGB")
    return image

def load_local_hf_model(model_path: str,
                  torch_dtype=torch.float32, 
                  device_map="cpu",
                  trust_remote_code=True,
                  **kwargs):
    """
    Load model and tokenizer from local path or HF model ID
    Args:
        model_path: Local path or HF model ID
    Returns:
        tuple: (model, tokenizer)
    """
    try:
        # Try loading as local path first
        if os.path.exists(model_path):
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
            	local_files_only=True,
            	trust_remote_code=trust_remote_code,
            	torch_dtype=torch_dtype,
            	device_map=device_map,
		        **kwargs
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=trust_remote_code,
                **kwargs
            )
        else:
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Please download the model first using:\n"
                f"huggingface-cli download --resume-download {model_path}"
            )
        
        return model, tokenizer
    
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

def prepare_inputs(processor, input_data, device=None, dtype=torch.float32, extra_args={}):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = processor(input_data, return_tensors="pt", **extra_args)
    inputs = {k: v.to(device).to(dtype) for k, v in inputs.items()}
    return inputs

def generate_text(model, tokenizer, prompt: str, device=None, dtype=torch.float32):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            pad_token_id=pad_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_modality(model, processor, input_data, device, dtype=torch.float32, extra_args={}):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    inputs = prepare_inputs(processor, input_data, device, dtype, extra_args)
    with torch.no_grad():
        out = model.generate(**inputs)

    return processor.decode(out[0], skip_special_tokens=True)

def get_generation(model, modality: str, input_data, device=None, dtype=torch.float32, tokenizer=None, processor=None):
    if modality == "text":
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided for text modality")
        return generate_text(model, tokenizer, input_data, device, dtype)
    elif modality in ("audio", "image", "video"):
        if processor is None:
            raise ValueError(f"Processor must be provided for {modality} modality")
        extra_args = {}
        if modality == "audio":
            extra_args["sampling_rate"] = 16000
        return generate_modality(model, processor, input_data, device, dtype, extra_args)
    else:
        raise ValueError(f"Unsupported modality: {modality}. Available: text, audio, image, video")

def get_generation_from_pipe(model_path: str):
    """
    Load a text-generation pipeline using a local model and tokenizer.
    """
    # Load tokenizer and model from local directory
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.float32,  # or torch.float32 for explicitness
        low_cpu_mem_usage=True
    )
    # Create the pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1  # -1 means CPU
    )
    return generator

def named_module_tensors(module, recurse=False):
    for named_parameter in module.named_parameters(re   curse=recurse):
        name, val = named_parameter
        flag = True
        if hasattr(val, "_data") or hasattr(val, "_scale"):
            if hasattr(val, "_data"):
                yield name + "._data", val._data
            if hasattr(val, "_scale"):
                yield name + "._scale", val._scale
        else:
            yield named_parameter

    for named_buffer in module.named_buffers(recurse=recurse):
        yield named_buffer

def dtype_byte_size(dtype):
    """
    Returns the size (in bytes) occupied by one parameter of type `dtype`.
    """
    import re
    
    if dtype == torch.bool:
        return 1/8
    
    bit_search = re.search(r"[^\d](\d+)$", str(dtype))
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid data type: {dtype}.")
    
    bit_size = int(bit_search.groups()[0])
    return bit_size / 8

def compute_model_sizes(model):
    """
    Compute the size of each submodule of a given model
    """
    from collections import defaultdict

    module_sizes = defaultdict(int)

    for name, tensor in named_module_tensors(module=model, recurse=True):
        size = tensor.numel() * dtype_byte_size(tensor.dtype)
        name_parts = name.split(".")
        for idx in range(len(name_parts)+1):
            module_sizes[".".join(name_parts[:idx])] += size
    
    return module_sizes

#-------------------------------------------------------#
#           Core Quantization Functions                  #
#-------------------------------------------------------#

def linear_q_with_scale_and_zero_point(tensor, scale, zero_point, dtype=torch.int8):
    """Performs linear quantization using scale and zero-point.
    
    Args:
        tensor: Input tensor to quantize
        scale: Scale factor for quantization
        zero_point: Zero point offset
        dtype: Target quantization type (default: torch.int8)
    
    Returns:
        Quantized tensor
    """
    scaled_tensor = tensor/scale + zero_point
    rounded_tensor = torch.round(scaled_tensor)
    q_min, q_max = torch.iinfo(dtype).min, torch.iinfo(dtype).max
    return rounded_tensor.clamp(min=q_min, max=q_max).to(dtype=dtype)

def linear_dequantization(quantized_tensor, scale, zero_point):
    """Performs linear dequantization using scale and zero-point.
    
    Args:
        quantized_tensor: Input quantized tensor
        scale: Scale factor used in quantization
        zero_point: Zero point offset used in quantization
    
    Returns:
        Dequantized floating point tensor
    """
    return scale * (quantized_tensor.float() - zero_point)

def get_q_scale_symmetric(tensor, dtype=torch.int8):
    """Calculate symmetric quantization scale.
    
    Args:
        tensor: Input tensor
        dtype: Target quantization type
    
    Returns:
        Quantization scale factor
    """
    r_max = tensor.abs().max().item()
    q_max = torch.iinfo(dtype).max
    return r_max/q_max

#-------------------------------------------------------#
#           Quantization Methods                         #
#-------------------------------------------------------#

def linear_q_symmetric_per_channel(r_tensor, dim, dtype=torch.int8):
    """Performs symmetric per-channel quantization.
    
    Args:
        r_tensor: Input tensor to quantize
        dim: Dimension along which to quantize
        dtype: Target quantization type
    
    Returns:
        tuple: (quantized_tensor, scale)
    """
    output_dim = r_tensor.shape[dim]
    scale = torch.zeros(output_dim)
    
    for index in range(output_dim):
        sub_tensor = r_tensor.select(dim=dim, index=index)
        scale[index] = get_q_scale_symmetric(sub_tensor, dtype=dtype)

    scale_shape = [1] * r_tensor.dim()
    scale_shape[dim] = -1
    scale = scale.view(scale_shape)

    quantized_tensor = linear_q_with_scale_and_zero_point(
        r_tensor, scale=scale, zero_point=0, dtype=dtype
    )
    return quantized_tensor, scale

#-------------------------------------------------------#
#           Forward Pass Functions                       #
#-------------------------------------------------------#

def w8_a32_forward(input, q_w, s_w, z_w=0, bias=None):
    """W8A32 forward pass implementation.
    
    Args:
        input: FP32 input tensor
        q_w: INT8 quantized weights
        s_w: Weight scales
        z_w: Zero point (default: 0 for symmetric quantization)
        bias: Optional bias tensor
    
    Returns:
        Output tensor
    """
    assert input.dtype == torch.float32
    assert q_w.dtype == torch.int8
    
    s_w = s_w.view(-1, 1)
    dequantized_weight = q_w.to(torch.float32) * s_w + z_w
    output = F.linear(input, dequantized_weight)
    
    if bias is not None:
        output += bias
    return output

def w8_a16_forward(weight, input, scales, bias=None):
    """W8A16 forward pass implementation.
    
    Args:
        weight: INT8 weights
        input: FP16 input tensor
        scales: Weight scales
        bias: Optional bias tensor
    
    Returns:
        Output tensor
    """
    casted_weights = weight.to(input.dtype)
    output = F.linear(input=input, weight=casted_weights) * scales
    
    if bias is not None:
        output += bias
    return output


