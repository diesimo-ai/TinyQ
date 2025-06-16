import os
import sys
import argparse
import logging
import time
import torch
import psutil
import gc
from pathlib import Path
from torchinfo import summary
from utils import load_model, setup_logging
from contextlib import contextmanager

@contextmanager
def track_memory():
    """Context manager to track memory usage"""
    try:
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 * 1024)
        yield
    finally:
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        mem_after = process.memory_info().rss / (1024 * 1024)
        print(f"Memory Change: {mem_after - mem_before:.2f} MB")

def get_hardware_info():
    """Get system hardware information"""
    cpu_count = psutil.cpu_count(logical=False)
    total_memory = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # GB
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "N/A"
    
    return {
        "cpu_cores": cpu_count,
        "total_memory": f"{total_memory:.2f}GB",
        "gpu_available": gpu_available,
        "gpu_name": gpu_name
    }

def optimize_torch_settings():
    """Apply PyTorch optimizations"""
    # Enable TF32 on Ampere GPUs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
    # Enable memory efficient attention if available
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        torch.backends.cuda.enable_flash_sdp(True)
    
    # Enable channels last memory format
    torch.backends.cudnn.benchmark = True

def run_inference_benchmark(model, tokenizer, logger, num_threads=None):
    """Run basic inference benchmark with thread optimization"""
    if num_threads:
        torch.set_num_threads(num_threads)
    
    sample_text = "Hello, world! This is a test prompt for benchmarking."
    warmup_rounds = 3
    test_rounds = 5
    latencies = []
    
    logger.info("Starting warmup rounds...")
    with torch.inference_mode(), track_memory():
        # Warmup rounds
        for _ in range(warmup_rounds):
            inputs = tokenizer(sample_text, return_tensors="pt")
            _ = model(**inputs)
    
        logger.info("Running benchmark rounds...")
        # Benchmark rounds
        for i in range(test_rounds):
            inputs = tokenizer(sample_text, return_tensors="pt")
            start_time = time.time()
            outputs = model(**inputs)
            latency = (time.time() - start_time) * 1000  # Convert to ms
            latencies.append(latency)
            logger.info(f"Round {i+1} latency: {latency:.2f}ms")
    
    return sum(latencies) / len(latencies)


def run_inference_text(model, tokenizer, input_text, logger, num_threads=None):
    """Run basic inference benchmark with thread optimization for text models
    
    Args:
        model: HuggingFace transformer model
        tokenizer: HuggingFace tokenizer
        input_text (str): Input text to benchmark
        logger: Logger instance
        num_threads (int, optional): Number of threads for optimization
    
    Returns:
        float: Average inference latency in milliseconds
    """
    if num_threads:
        torch.set_num_threads(num_threads)
    
    # Use provided text or fallback to default
    sample_text = input_text if input_text else "Hello, world! This is a test prompt for benchmarking."
    warmup_rounds = 3
    test_rounds = 5
    latencies = []
    
    logger.info("Starting warmup rounds...")
    with torch.inference_mode(), track_memory():
        # Warmup rounds
        for _ in range(warmup_rounds):
            inputs = tokenizer(sample_text, return_tensors="pt")
            _ = model(**inputs)
    
        logger.info("Running benchmark rounds...")
        # Benchmark rounds
        for i in range(test_rounds):
            inputs = tokenizer(sample_text, return_tensors="pt")
            start_time = time.time()
            outputs = model(**inputs)
            latency = (time.time() - start_time) * 1000  # Convert to ms
            latencies.append(latency)
            logger.info(f"Round {i+1} latency: {latency:.2f}ms")
    
    return sum(latencies) / len(latencies)

def run_inference_audio(model, processor, audio_path, logger, num_threads=None):
    """Run basic inference benchmark with thread optimization for audio models
    
    Args:
        model: HuggingFace transformer model
        processor: HuggingFace audio processor
        audio_path (str): Path to audio file
        logger: Logger instance
        num_threads (int, optional): Number of threads for optimization
    
    Returns:
        float: Average inference latency in milliseconds
    """
    if num_threads:
        torch.set_num_threads(num_threads)
    
    # Process audio input
    try:
        if audio_path:
            audio_input = processor(audio_path, sampling_rate=16000, return_tensors="pt")
        else:
            # Create a synthetic audio signal for testing
            sample_rate = 16000
            duration = 2  # seconds
            t = torch.linspace(0, duration, int(sample_rate * duration))
            audio_input = torch.sin(2 * torch.pi * 440 * t)  # 440 Hz sine wave
            audio_input = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt")
    except Exception as e:
        logger.error(f"Error processing audio input: {e}")
        return None
    
    warmup_rounds = 3
    test_rounds = 5
    latencies = []
    
    logger.info("Starting warmup rounds...")
    with torch.inference_mode(), track_memory():
        # Warmup rounds
        for _ in range(warmup_rounds):
            _ = model(**audio_input)
    
        logger.info("Running benchmark rounds...")
        # Benchmark rounds
        for i in range(test_rounds):
            start_time = time.time()
            outputs = model(**audio_input)
            latency = (time.time() - start_time) * 1000  # Convert to ms
            latencies.append(latency)
            logger.info(f"Round {i+1} latency: {latency:.2f}ms")
    
    return sum(latencies) / len(latencies)

def run_inference_vision(model, processor, image_path, logger, num_threads=None):
    """Run basic inference benchmark with thread optimization for vision models
    
    Args:
        model: HuggingFace transformer model
        processor: HuggingFace image processor
        image_path (str): Path to image file
        logger: Logger instance
        num_threads (int, optional): Number of threads for optimization
    
    Returns:
        float: Average inference latency in milliseconds
    """
    if num_threads:
        torch.set_num_threads(num_threads)
    
    # Process image input
    try:
        if image_path:
            from PIL import Image
            image = Image.open(image_path)
            inputs = processor(images=image, return_tensors="pt")
        else:
            # Create a test image (black and white checkerboard)
            import numpy as np
            test_image = np.zeros((224, 224, 3), dtype=np.uint8)
            test_image[::2, ::2] = 255  # Create checkerboard pattern
            inputs = processor(images=test_image, return_tensors="pt")
    except Exception as e:
        logger.error(f"Error processing image input: {e}")
        return None
    
    warmup_rounds = 3
    test_rounds = 5
    latencies = []
    
    logger.info("Starting warmup rounds...")
    with torch.inference_mode(), track_memory():
        # Warmup rounds
        for _ in range(warmup_rounds):
            _ = model(**inputs)
    
        logger.info("Running benchmark rounds...")
        # Benchmark rounds
        for i in range(test_rounds):
            start_time = time.time()
            outputs = model(**inputs)
            latency = (time.time() - start_time) * 1000  # Convert to ms
            latencies.append(latency)
            logger.info(f"Round {i+1} latency: {latency:.2f}ms")
    
    return sum(latencies) / len(latencies)

def main():
    parser = argparse.ArgumentParser(description="Benchmark script for model quantization.")
    parser.add_argument("--model_path", type=str, required=True,
                      help="Path to local model directory")
    parser.add_argument("--log_dir", type=str, default="logs",
                      help="Directory for all logs")
    parser.add_argument("--num_threads", type=int,
                      help="Number of threads for inference")
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging("benchmark", args.log_dir)
    
    # Get and log hardware info
    hw_info = get_hardware_info()
    logger.info("\nHardware Configuration:")
    for key, value in hw_info.items():
        logger.info(f"{key}: {value}")
    
    # Apply optimizations
    optimize_torch_settings()
    
    try:
        with track_memory():
            # Load model and tokenizer with optimizations
            logger.info(f"\nLoading model from {args.model_path}")
            model, tokenizer = load_model(
                args.model_path,
                device_map='auto',  # Auto-detect best device
                torch_dtype=torch.float32 if not torch.cuda.is_available() else torch.float16,
                low_cpu_mem_usage=True
            )
            
            # Set optimal number of threads if not specified
            if not args.num_threads:
                args.num_threads = psutil.cpu_count(logical=False)
            
            # Run benchmark
            logger.info("\nRunning Inference Benchmark:")
            avg_latency = run_inference_benchmark(model, tokenizer, logger, args.num_threads)
            logger.info(f"\nAverage Inference Latency: {avg_latency:.2f}ms")
            
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()