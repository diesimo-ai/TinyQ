# Quantizer

A lightweight PyTorch quantization library focusing on post-training static quantization for efficient CPU deployment.

## Features

* **Post-Training Quantization (PTQ) Focus:** Quantization is applied after model training
* **Quantization Schemes:** Linear quantization (symmetric/asymmetric)
* **Granular Methods:** Per-channel quantization for weights, per-tensor for activations
* **Supported Bit-Widths:** W8A32 and W8A16 precision
* **Model Support:** Works with PyTorch models, especially from Hugging Face Hub

## Installation

```bash
git clone https://github.com/afondiel/Quantizer.git
cd Quantizer
pip install -r requirements.txt
```

## Quick Start

```python
from quantizer import Quantizer
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True,
                       help='Path to local model or Hugging Face model ID')
    parser.add_argument('--q_method', default='w8a32',
                       choices=['w8a32', 'w8a16'])
    args = parser.parse_args()

    quantizer = Quantizer()
    quantizer.load_model(args.model_path)
    quantizer.quantize(q_method=args.q_method)
    quantizer.save_model("./quantized_model")

if __name__ == "__main__":
    main()
```

Run quantization:
```bash
python examples.py --model_path "Salesforce/codegen-350M-mono" --q_method w8a32
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

## Contributing

Contributions are welcome! If you want to help this project grow, you can pick one of the listed topics in the [Roadmap](#roadmap). Please see the [Contributing Guidelines](CONTRIBUTING.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.