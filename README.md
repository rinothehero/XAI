# Project: Model Pruning, Quantization, and Analysis for BERT on AG News

This project explores various techniques to optimize a BERT model fine-tuned on the AG News dataset. It includes scripts for:

- Fine-tuning the base BERT model.
- Applying structural pruning and dynamic quantization.
- Exporting models to ONNX format (including structurally shrunk versions).
- Generating LIME (Local Interpretable Model-agnostic Explanations) for model behavior analysis.
- Comparing LIME explanations of the baseline model with optimized versions.
- Training simple MLP models to mimic BERT's explanations (experimental).
- Running inference and benchmarking different model versions.

## Project Structure

Here's an overview of the key directories:

-   **`models/`**: (Gitignored) Output directory for all generated models.
    -   `fine_tuned/bert_agnews/`: Base fine-tuned BERT model.
    -   `pruned/bert_agnews_Xpct/`: Pruned (and then re-fine-tuned) models.
    -   `quantized/`: Quantized models (e.g., `bert_agnews_8bit.pt`).
    -   `shrunk/`: Structurally shrunk PyTorch models.
    -   `onnx/`: Exported ONNX models.
-   **`outputs/`**: (Gitignored) Output directory for generated data, reports, and intermediate files.
    -   `metrics/`: Inference summaries and performance metrics.
    -   `reports/`: Final analysis reports (e.g., LIME similarity).
    -   `intermediate_data/`: Intermediate files like token explanations for MLP training.
    -   `sp_lime_sets/`: Output from SP-LIME (Submodular Pick LIME) analysis.
-   **`analysis/`**: Scripts for analyzing models.
    -   `monitoring/`: CPU/GPU memory and performance monitoring scripts.
    -   `model_properties/`: Scripts to check model size, sparsity, parameters.
    -   `nltk/`: Scripts for NLTK-based POS tagging analysis of LIME explanations (`nltk_analyze_all_models.py`).
-   **`export_onnx/`**: Scripts for exporting models to ONNX.
    -   `shrinking_utils.py`: Utility functions for structurally shrinking models.
    -   `export-onnx-prune-20.py`, `export-onnx-prune-30.py`: Export structurally shrunk pruned models.
    -   `onnx-pruned.py`: Exports a pruned model without structural shrinking.
    -   `save_shrunk_pytorch_model.py`: Applies structural shrinking and saves the PyTorch model.
-   **`generate_outputs/`**: Scripts for generating various analysis outputs.
    -   `generate_lime_csv/`:
        -   `run_lime_on_model.py`: (Placeholder) Intended for generating LIME explanation CSVs for various models.
        -   `generate_bert_token_explanations_for_mlp.py`: Generates detailed token-level LIME explanations from BERT, used as input for MLP training.
        -   `run_sp_lime_hf_model.py`, `run_sp_lime_torch_loaded_model.py`: Scripts for SP-LIME analysis.
    -   `lime_similarity_pipeline/`: Scripts to compare LIME explanations of different models.
        -   `run_pipeline.sh`: Executes the full LIME comparison pipeline.
-   **`inference/`**: Scripts for running inference and benchmarks.
    -   `run_inference_on_model.py`: (Placeholder) Intended for running inference with various models and collecting metrics.
    -   `evaluate_agnews_quantized_16bit.py`: Evaluates a 16-bit quantized model on the AG News test set.
    -   `run_onnx_benchmark.py`: Benchmarks ONNX model inference speed.
    -   `inference.sh`: Example script to run inference with selected models and monitors.
-   **`train/`**: Scripts for training models.
    -   `fine_tuning/run_fine_tune_bert_agnews.py`: Fine-tunes the base BERT model.
    -   `pruning/run_structural_pruning.py`: (Placeholder) Intended for applying structured pruning and re-fine-tuning.
    -   `quantization/run_dynamic_quantization.py`: (Placeholder) Intended for applying dynamic quantization.
-   **`train_mlp_models/`**: Scripts for training MLP models.
    -   `train_mlp_vanilla.py`: Trains an MLP with standard cross-entropy loss.
    -   `train_mlp_xai_loss.py`: Trains an MLP using an XAI-guided loss function (to match BERT LIME explanations).
-   **`tools/`**: Utility scripts.
    -   `apply_shrink_to_model.py`: Applies structural shrinking to a Hugging Face model and saves it.
-   **`.gitignore`**: Specifies intentionally untracked files.

## Core Workflows

1.  **Fine-tune Base Model:**
    -   Run `train/fine_tuning/run_fine_tune_bert_agnews.py`.
    -   Model saved to `models/fine_tuned/bert_agnews/`.

2.  **Pruning (Example):**
    -   (Implement `train/pruning/run_structural_pruning.py` to take fine-tuned model path, pruning params, and output path like `models/pruned/bert_agnews_20pct/`)
    -   This script should apply pruning and then re-fine-tune the model.

3.  **Quantization (Example):**
    -   (Implement `train/quantization/run_dynamic_quantization.py` to take a model path, quantization params, and output path like `models/quantized/bert_agnews_8bit.pt`)

4.  **Structural Shrinking & Export:**
    -   To shrink a HuggingFace compatible pruned model and save it as a PyTorch model:
        `python tools/apply_shrink_to_model.py --model_path <path_to_pruned_hf_model> --output_path_shrunk_model <models/shrunk/my_shrunk_model.pt_or_dir> --save_method [torch_save|save_pretrained]`
    -   To export a shrunk model to ONNX (after loading and shrinking or loading an already shrunk one):
        Use scripts in `export_onnx/` like `export-onnx-prune-20.py` (update its input model path).

5.  **LIME Explanations & Analysis:**
    -   Generate base explanations for MLP training: `python generate_outputs/generate_lime_csv/generate_bert_token_explanations_for_mlp.py` (outputs to `outputs/intermediate_data/`).
    -   Generate LIME CSVs for specific models using the (to be implemented) `generate_outputs/generate_lime_csv/run_lime_on_model.py`.
    -   Run SP-LIME: `python generate_outputs/generate_lime_csv/run_sp_lime_hf_model.py --model_dir models/fine_tuned/bert_agnews --output_prefix outputs/sp_lime_sets/base_model_sp_lime`
    -   Compare LIME explanations: `cd generate_outputs/lime_similarity_pipeline/ && ./run_pipeline.sh` (ensure input paths in scripts are correct).

6.  **Inference & Benchmarking:**
    -   Run `inference/inference.sh` (after updating it to use `run_inference_on_model.py`).
    -   Evaluate specific models: e.g., `python inference/evaluate_agnews_quantized_16bit.py`.
    -   Benchmark ONNX: `python inference/run_onnx_benchmark.py`.

## Dependencies

Key Python libraries used:
-   `transformers`
-   `torch`
-   `datasets`
-   `evaluate`
-   `lime`
-   `onnx`
-   `onnxruntime`
-   `pandas`
-   `numpy`
-   `scikit-learn` (for SP-LIME cosine similarity)
-   `scipy` (for Spearman correlation)

(A `requirements.txt` file should ideally be generated and maintained.)

## Notes on Placeholders

Scripts marked as "(Placeholder)" or "(To be implemented)" require further development to consolidate logic from the older, more specific scripts that were removed during refactoring. The placeholders indicate the intended new location and purpose of these consolidated scripts. The refactoring focused on structure and removing direct redundancy, but fully implementing these generic, parameterized scripts is the next step for code usability.
Paths inside many scripts (especially for loading models and data) have been updated to reflect the new structure but always verify them before running.
