# Model Setup Guide

TinyQ operates in offline-only mode. Before using the library, you need to download your models locally.

## Downloading Hugging Face Models

1. Install the Hugging Face CLI:
```bash
pip install --upgrade huggingface-hub
```

2. Download your model:
```bash
# Example for CodeGen model
huggingface-cli download --resume-download Salesforce/codegen-350M-mono --local-dir ./models/Salesforce/codegen-350M-mono
```

3. Verify the model structure:
```
models/
└── Salesforce/
    └── codegen-350M-mono/
        ├── config.json
        ├── pytorch_model.bin
        ├── tokenizer.json
        └── tokenizer_config.json
```

## Model Directory Structure
TinyQ expects models to be organized as follows:
```
models/
├── huggingface/        # For HuggingFace models
├── torchvision/        # For torchvision models (coming soon)
└── custom/            # For custom PyTorch models (coming soon)
```

## Usage Example
```python
from tinyq import Quantizer

# Use local path
model_path = "./models/Salesforce/codegen-350M-mono"
quantizer = Quantizer(model_path, local_only=True)
```