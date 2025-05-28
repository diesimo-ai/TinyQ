import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from utils import (linear_q_symmetric_per_channel, 
                  w8_a32_forward, 
                  w8_a16_forward)

class W8A32LinearLayer(nn.Module):
    """
    Custom Linear Module with 8-bit weights and 32-bit activations (W8A32).
    """
    def __init__(self, in_features, out_features, bias=True, dtype=torch.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer("int8_weights",
                              torch.randint(low=-128,
                                            high=127,
                                            size=(out_features, in_features),
                                            dtype=torch.int8))
        self.register_buffer("scales",
                             torch.randn((out_features), dtype=dtype)) 
        self.register_buffer("zero_points",
                             torch.zeros((out_features), dtype=dtype)) 

        if bias:
            self.register_buffer("bias",
                                 torch.randn((1, out_features), dtype=dtype))
        else:
            self.bias = None

    def quantize(self, weights):
        """
        Quantizes the input FP32 weights to INT8 and stores them along with
        their quantization parameters.

        Args:
            weights (torch.Tensor): The original FP32 weight tensor.
        """
        w_fp32 = weights.clone().to(torch.float32)
        int8_weights, scales = linear_q_symmetric_per_channel(w_fp32, dim=0, dtype=torch.int8)

        self.int8_weights = int8_weights
        self.scales = scales.squeeze() 
        self.zero_points = torch.zeros_like(self.scales, dtype=torch.int32) 

    def forward(self, input):
        return w8_a32_forward(input=input, q_w=self.int8_weights, s_w=self.scales, z_w=0, bias=self.bias)

# Custom Linear Module: W8A16
class W8A16LinearLayer(nn.Module):
    """Custom Linear Module with 8-bit weights and 16-bit activations (W8A16).
    
    Implements weight quantization to INT8 and activation handling in FP16.
    """    
    
    def __init__(self, in_features, out_features, bias=True, dtype=torch.float32):
        super().__init__()

        self.register_buffer("int8_weights",
                              torch.randint(low=-128,
                                            high=127,
                                            size=(out_features, in_features),
                                            dtype=torch.int8)
                            )

        self.register_buffer("scales",
                             torch.randn((out_features), dtype=dtype))

        if bias:
            self.register_buffer("bias",
                                 torch.randn((1, out_features), dtype=dtype))
        else:
            self.bias = None

    def quantize(self, weights):
        w_fp32 = weights.clone().to(torch.float32)

        scales = w_fp32.abs().max(dim=-1).values / 127
        scales = scales.to(weights.dtype)

        int8_weights = torch.round(weights/scales.unsqueeze(1)).to(torch.int8)

        self.int8_weights = int8_weights
        self.scales = scales

    def forward(self, input):
        return w8_a16_forward(weight=self.int8_weights, input=input, scales=self.scales, bias=self.bias)

# Custom Linear Module: W8A8
class W8A8LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        pass 

def replace_linear_with_target(module, target_class, module_name_to_exclude):
    """
    Replaces nn.Linear layers with instances of target_class.

    Args:
        module (nn.Module): The module to modify.
        target_class (type): The class to replace nn.Linear with.
        module_name_to_exclude (list): List of module names to exclude from replacement.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and not any([x==name for x in module_name_to_exclude]):
            old_bias = child.bias
            new_module = target_class(child.in_features,
                                      child.out_features,
                                      bias=old_bias is not None,
                                      dtype=child.weight.dtype)
            setattr(module, name, new_module)

            if old_bias is not None:
                getattr(module, name).bias = old_bias
        else:
            # Recursively call the function for nested modules
            replace_linear_with_target(child, target_class, module_name_to_exclude)

# Replace Linear Layer Module and Quantize
def replace_linear_with_target_and_quantize(module, target_class, module_name_to_exclude):
    """
    Replaces nn.Linear layers with instances of target_class and quantizes weights.
    Handles different target_class __init__ signatures.
    Args:
        module (nn.Module): The module to modify.
        target_class (type): The class to replace nn.Linear with.
        module_name_to_exclude (list): List of module names to exclude from replacement.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and not any([x==name for x in module_name_to_exclude]):
            old_bias = child.bias
            old_weight = child.weight

            # Dynamically create the new module based on target_class signature
            try:
                # Try creating with in_features, out_features, bias, and dtype (for W8A16)
                new_module = target_class(child.in_features,
                                          child.out_features,
                                          bias=old_bias is not None,
                                          dtype=child.weight.dtype)
            except TypeError:
                # If dtype argument is not accepted (for W8A32)
                new_module = target_class(child.in_features,
                                          child.out_features,
                                          bias=old_bias is not None)


            setattr(module, name, new_module)
            getattr(module, name).quantize(old_weight)

            if old_bias is not None:
                getattr(module, name).bias = old_bias

        else:
            # Recursively call for nested modules
            replace_linear_with_target_and_quantize(child, target_class, module_name_to_exclude)


#-------------------------------------------------------#
#           Quantizer Workflow                          #
#-------------------------------------------------------#

class Quantizer:
    def __init__(self, model: nn.Module):
        """
        Initialize quantizer with a pre-loaded model
        Args:
            model: PyTorch model to quantize
        """
        self.model = model
        self.quantized_model = None
    
    def quantize(self, q_method='w8a32'):
        """
        Quantize the model using specified method
        Args:
            q_method: Quantization method ('w8a32' or 'w8a16')
        Returns:
            nn.Module: Quantized model
        """
        self.q_method = q_method
        self.module_name_to_exclude = module_not_to_quantize or []

        if self.q_method not in ["w8a32", "w8a16"]:
            raise ValueError("Supported methods: 'w8a32', 'w8a16'")

        target_class = W8A32LinearLayer if q_method == "w8a32" else W8A16LinearLayer
        print(f"Applying {q_method} quantization...")
        replace_linear_with_target_and_quantize(
            self.model, 
            target_class, 
            self.module_name_to_exclude
        )
        print(f"{q_method} quantization applied.")
        return self.quantized_model
    
    def save_model(self, save_path: str):
        """Save quantized model"""
        if self.quantized_model is None:
            raise ValueError("No quantized model available. Run quantize() first.")
        torch.save(self.quantized_model.state_dict(), save_path)
