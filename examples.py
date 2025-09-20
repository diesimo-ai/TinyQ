import os
import argparse
import torch
from tinyq import Quantizer
from utils import load_local_hf_model, get_generation, \
                  get_generation_from_pipe, setup_logging, \
                  parse_args

def main(args):
    
    # Setup main logger
    logger = setup_logging("tinyq", "logs")
    logger.info("TinyQ session started")
    
    try:
        # Load model and tokenizer
        logger.info(f"Loading model from {args.model_path}")
        model, tokenizer = load_local_hf_model(
            args.model_path,
            device_map='cpu',
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

        # Display original model info
        print("\n" + "="*50)
        print("ORIGINAL MODEL ARCHITECTURE")
        print("="*50)
        print(model)

        # Initialize quantizer with main logger
        q = Quantizer(logger=logger)
        
        # Apply quantization
        logger.info(f"Quantizing model using {args.qm}")
        qmodel = q.quantize(
            model,
            q_method=args.qm,
            module_not_to_quantize=['lm_head']
        )

        # Display quantized model info
        print("\n" + "="*50)
        print(f"QUANTIZED MODEL ARCHITECTURE ({args.qm})")
        print("="*50)
        print(qmodel)
        
        # Optional: Test model outputs
        if args.test_inference:
            prompt = "Hello, my name is"
            logger.info("Running inference test")
            with torch.inference_mode():
                result = get_generation(
                    qmodel,
                    "text",
                    input_data=prompt,
                    tokenizer=tokenizer,
                    max_new_tokens=10
                )
            print("\nTest output:", result)
        
        # Save quantized model
        logger.info(f"Saving quantized model to {args.qmodel_path}")
        q.export(args.qmodel_path, qmodel)
        print(f"Model saved to {args.qmodel_path}")

        logger.info("TinyQ session completed successfully")
        
    except Exception as e:
        logger.error(f"Error during quantization: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quantize PyTorch models')
    parser.add_argument('--model_path', type=str, required=True, help='Path to local model directory')
    parser.add_argument('--qmodel_path', type=str, default='./qmodel', help='Directory to save quantized model')
    parser.add_argument('--qm', type=str, default='w8a32', choices=['w8a32', 'w8a16'], help='Quantization method to use')
    parser.add_argument('--test_inference', action='store_true', help='Run inference test after quantization')

    args = parser.parse_args()

    main(args)
