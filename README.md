# TinyQ

A compact, simple and efficient post-training quantization module built on the top of PyTorch's Modules. 

## Features

- **PTQ Focus**: quantize of all Linear Layers (nn.Linear)
- **Quantization Methods**: `W8A32` (8-bit weights, 32-bit activations), `W8A16` (8-bit weights, 16-bit activations), `W8A8` (Coming soon!)
- **Model Support**: PyTorch models from [Hugging Face Hub](https://huggingface.co/models?library=pytorch)
- **Offline-first approach**: no automatic downloads from the cloud
- **Built-in benchmarking**: Memory footprint vs latency tracking (Coming soon)

## Installation

```bash
git clone https://github.com/yourusername/TinyQ.git
cd TinyQ

# Create and activate conda environment
conda create -n tinyq python=3.8
conda activate tinyq

# Install requirements
pip install -r requirements.txt
```

## Quick Start

### 1. Download a Model

> [!IMPORTANT]
> This current version operates in `offline-only` mode. Download your model first:

```bash
# Example: Download OPT-125M
huggingface-cli download --resume-download facebook/opt-125m --local-dir ./models/facebook/opt-125m
```

See the full [Model Setup Guide](docs/model_setup.md) for detailed instructions.

### 2. Run Quantization

```python
from tinyq import Quantizer
from utils import load_model, get_generation

# Load model
model, tokenizer = load_model("./models/facebook/opt-125m")

# Initialize quantizer
quantizer = Quantizer(model)

# Quantize model (W8A32 or W8A16)
quantized_model = quantizer.quantize(q_method="w8a32")

# Test inference
prompt = "Hello, world!"
result = get_generation(quantized_model, tokenizer, prompt)
print(result)
```

### 3. Run Benchmark (On going)

```bash
python bench.py --model_path "./models/facebook/opt-125m"
```

## Command Line Interface (CLI) 

```bash
python examples.py \
    --model_path "./models/facebook/opt-125m" \
    --qm w8a32 \
    --test_inference \
    --qmodel_path "./quantized_model"
```

Arguments:
- `--model_path`: Path to local model directory (required)
- `--qm`: Quantization method [`w8a32`, `w8a16`] (default: w8a32)
- `--test_inference`: Run inference test after quantization
- `--qmodel_path`: Save path for quantized model (default: ./quantized_model)

## Project Structure 

```
TinyQ/
├── logs/              # Benchmark and training logs
├── models/            # Local model storage
├── tinyq.py           # Core quantization library
├── utils.py           # Utility functions
├── examples.py        # Usage examples
└── bench.py           # Benchmarking tools (Coming soon)
```

## Roadmap

### Current Focus
- [x] W8A32 implementation
- [x] W8A16 implementation
- [ ] Documentation and examples
- [ ] Unit tests

### Core Features
- [ ] W8A8 Quantization Support
- [ ] Model Support Extensions
- [ ] Additional Layer Support
- [ ] Performance Optimization

## Demo

The example below shows a Pytorch model printout before and after applying W8A32 Quantization.

Before: 

![](./demo/model-before.png)

After:

![](./demo/model-after.png)

You can also use a tool like [NEUTRON](https://netron.app/) get more in-depth insight and compare both models.

## Benchmark

(Still to Come)

## Contributing 

Contributions are welcome! Please see the [Contributing Guidelines](CONTRIBUTING.md).

## License 

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project started as a learning exercise from the [Quantization Fundamentals](https://www.deeplearning.ai/short-courses/quantization-fundamentals-with-hugging-face/) course by DeepLearning.AI and Hugging Face, helping me understand the core concepts behind model quantization.

Special thanks to:
- Younes Belkada & Marc Sun for their excellent instruction and course content
- Andrew Ng and the DeepLearning.AI team for making AI education accessible and practical
- [kaushikacharya](https://github.com/kaushikacharya) for their detailed course notes that provided valuable guidance

