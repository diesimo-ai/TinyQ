import argparse
from tinyq import Quantizer
from utils import load_model, get_generation

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Quantize PyTorch models')
    parser.add_argument(
        '--model_path', 
        type=str, 
        required=True,
        help='Path to local model directory'
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
    
    # Load model and tokenizer
    model, tokenizer = load_model(args.model_path)
    
    # Optional: Test model before quantization
    prompt = "This is a test prompt"
    result = get_generation(model, tokenizer, prompt)
    print(f"Original model output: {result}")
    
    # Initialize quantizer with loaded model
    quantizer = Quantizer(model)
    
    # Apply quantization
    quantized_model = quantizer.quantize(q_method=args.qm)
    
    # Test quantized model
    result = get_generation(quantized_model, tokenizer, prompt)
    print(f"Quantized model output: {result}")
    
    # Save quantized model
    quantizer.save_model(args.qmodel_path)

if __name__ == "__main__":
    main()