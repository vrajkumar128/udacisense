"""
UdaciSense Project: Post-Training Quantization Module

This module provides utilities for applying post-training quantization to PyTorch models,
supporting both static and dynamic quantization methods.
"""

import os
import copy
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.ao.quantization.quantize_fx as quantize_fx
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.ao.quantization import QuantStub, DeQuantStub, fuse_modules

# TODO: Make MobileNetV3_Household model quantizable using stubs
# Consider whether you want to quantize the whole model or parts of it only
class QuantizableMobileNetV3_Household(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.quant = QuantStub()
        self.model = original_model
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x
    
    def fuse_model(self) -> None:
        """
        Fuse conv, bn, relu layers for better quantization results

        Args:
            model: Model to fuse
        """
        print("Fusing layers...")

        # TODO: Identify patterns to fuse (Conv+BN, Conv+BN+ReLU, etc.)
        for module in self.model.modules():
            if not isinstance(module, nn.Sequential):
                continue

            layers = list(module)

            if len(layers) >= 2 and isinstance(layers[0], nn.Conv2d) and isinstance(layers[1], nn.BatchNorm2d):
                if len(layers) >= 3 and isinstance(layers[2], nn.ReLU):
                    fuse_modules(module, ["0", "1", "2"], inplace=True)
                else:
                    fuse_modules(module, ["0", "1"], inplace=True)
        

def quantize_model(
    model: nn.Module,
    calibration_data_loader: Optional[DataLoader] = None,
    calibration_num_batches: Optional[int] = None,
    quantization_type: str = "dynamic",
    backend: str = "fbgemm",
) -> nn.Module:
    """Apply post-training quantization to a PyTorch model.
    
    Args:
        model: The original model to quantize
        calibration_data_loader: DataLoader for calibration data,
            required for static quantization
        calibration_num_batches: Number of batches to run calibration on
        quantization_type: Type of quantization to apply:
            - "dynamic": Dynamic quantization (weights are quantized, activations quantized during inference)
            - "static": Static quantization (weights and activations are pre-quantized)
        backend: Quantization backend, either "fbgemm" (x86) or "qnnpack" (ARM)
            
    Returns:
        Quantized model
        
    Raises:
        ValueError: If an unsupported backend or quantization type is specified,
                   or if static quantization is requested without calibration data
    """
    # Verify backend
    if backend not in ["fbgemm", "qnnpack"]:
        raise ValueError("Backend must be either 'fbgemm' (x86) or 'qnnpack' (ARM)")
    
    # Create a copy of the model for quantization
    model_to_quantize = copy.deepcopy(model)
    
    # Set model to evaluation mode
    model_to_quantize.eval()
    
    # NOTE: Feel free to not implement all quantization types
    # Apply quantization based on type
    if quantization_type.lower() == "dynamic":
        return _apply_dynamic_quantization(model_to_quantize)
    elif quantization_type.lower() == "static":
        if calibration_data_loader is None:
            raise ValueError("Static quantization requires a calibration_data_loader")
        return _apply_static_quantization(model_to_quantize, calibration_data_loader, calibration_num_batches, backend)
    else:
        raise ValueError(f"Unsupported quantization type: {quantization_type}")

# TODO: Implement dynamic quantization, if selected
# Remember to look at built-in pytorch functionalities whenever possible
def _apply_dynamic_quantization(
    model: nn.Module
) -> nn.Module:
    """Apply dynamic quantization to a model.
    
    Dynamic quantization quantizes weights ahead of time but quantizes activations
    dynamically during inference.
    
    Args:
        model: Model to quantize (in eval mode)
        
    Returns:
        Dynamically quantized model
    """
    return torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )
                

# TODO: Implement static quantization, if selected
# Remember to look at built-in pytorch functionalities whenever possible
# And that you first need to prepare the model for quantization, then apply calibration, and finally convert the model to quantized
def _apply_static_quantization(
    model: nn.Module,
    calibration_data_loader: DataLoader,
    calibration_num_batches: Optional[int] = None,
    backend: str = "fbgemm",
) -> nn.Module:
    """Apply static quantization to a model using provided calibration data.
    
    Static quantization quantizes both weights and activations ahead of time.
    
    Args:
        model: Model to quantize (in eval mode)
        calibration_data_loader: DataLoader for calibration data
        calibration_num_batches: Number of batches to use for calibration
        backend: Quantization backend, either "fbgemm" (x86) or "qnnpack" (ARM)
        
    Returns:
        Statically quantized model
    """
    print("Applying static quantization...")
    
    # If calibration_num_batches is not specified, use all available batches
    if calibration_num_batches is None:
        calibration_num_batches = len(calibration_data_loader)
        
    torch.backends.quantized.engine = backend
    qconfig_mapping = torch.ao.quantization.get_default_qconfig_mapping(backend)

    example_inputs = next(iter(calibration_data_loader))[0]
    prepared_model = quantize_fx.prepare_fx(model, qconfig_mapping, example_inputs)

    with torch.no_grad():
        for i, batch in enumerate(calibration_data_loader):
            if i >= calibration_num_batches:
                break
            inputs = batch[0]
            prepared_model(inputs)

    return quantize_fx.convert_fx(prepared_model)