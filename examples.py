import argparse
from quantizer import Quantizer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Quantize PyTorch models')
    parser.add_argument(
        '--model_path', 
        type=str, 
        required=True,
        help='Path to local model directory or Hugging Face model ID'
    )
    parser.add_argument(
        '--qmodel_path', 
        type=str, 
        default='./quantized_model',
        help='Directory to save quantized model'
    )
    parser.add_argument(
        '--qm', 
        type=str, 
        default='w8a32',
        choices=['w8a32', 'w8a16'],
        help='Quantization method to use'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize quantizer
    quantizer = Quantizer()
    
    # Load model
    quantizer.load_model(args.model_path)
    
    # Apply quantization
    quantizer.quantize(q_method=args.qm)
    
    # Save quantized model
    quantizer.save_model(args.qmodel_path)

if __name__ == "__main__":
    main()