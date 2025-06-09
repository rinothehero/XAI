import torch
from transformers import AutoModelForSequenceClassification # Ensure AutoTokenizer is also imported if needed by model loading
from export_onnx.shrinking_utils import apply_structural_shrink # Path relative to project root
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a pruned model, apply structural shrinking, and save the shrunk model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-pruned Hugging Face model directory.")
    parser.add_argument("--output_path_shrunk_model", type=str, required=True, help="Path to save the shrunk PyTorch model (.pt file or directory for save_pretrained).")
    parser.add_argument("--save_method", type=str, default="torch_save", choices=["torch_save", "save_pretrained"], help="Method to save the model.")

    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.eval()

    print("Applying structural shrink...")
    model = apply_structural_shrink(model)

    # Ensure output directory exists if saving to a nested path
    output_dir = os.path.dirname(args.output_path_shrunk_model)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if args.save_method == "torch_save":
        # Note: torch.save saves the whole model object. For just state_dict, use model.state_dict()
        torch.save(model, args.output_path_shrunk_model)
        print(f"✅ Shrunk model saved using torch.save to: {args.output_path_shrunk_model}")
    else: # save_pretrained
        model.save_pretrained(args.output_path_shrunk_model)
        # If tokenizer needs to be saved with it, it should be loaded and saved here too.
        # tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        # tokenizer.save_pretrained(args.output_path_shrunk_model)
        print(f"✅ Shrunk model saved using save_pretrained to: {args.output_path_shrunk_model}")
