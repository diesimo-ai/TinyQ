import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

#-------------------------------------------------------#
#           Preprocessing utilities                     #
#-------------------------------------------------------#
def load_model(model_path: str):
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
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True
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

def get_generation(model, tokenizer, prompt: str, dtype=torch.float32):
    """
    Run inference on the model
    Args:
        model: PyTorch model
        tokenizer: Model tokenizer  
        prompt: Input text
        dtype: Model dtype for inference
    Returns:
        str: Generated text
    """
    model.eval()
    model = model.to(dtype=dtype)
    
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=50)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return result

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


