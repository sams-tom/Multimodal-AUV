import os
import logging
import torch
from torch.utils.data import DataLoader, ConcatDataset
from collections import OrderedDict
import pandas as pd

from Multimodal_AUV.models.model_utils import define_models
from Multimodal_AUV.inference.predictors import multimodal_predict_and_save
from Multimodal_AUV.data.datasets import CustomImageDataset_1

def prepare_inference_dataloader(data_dir: str, batch_size: int) -> DataLoader:
    """
    Prepares an inference DataLoader for a single dataset directory.
    Uses your actual CustomImageDataset_1.
    """
    try:
        if not os.path.exists(data_dir):
            logging.error(f"Dataset directory not found: {data_dir}")
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

        dataset = CustomImageDataset_1(data_dir)
        if len(dataset) == 0:
            logging.error(f"No samples found in CustomImageDataset_1 from {data_dir}. Check your dataset implementation and directory contents.")
            raise ValueError(f"No samples found in dataset from {data_dir}")

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        logging.info(f"DataLoader prepared for '{data_dir}' with {len(dataset)} items.")
        return dataloader
    except Exception as e:
        logging.error(f"Error preparing inference DataLoader from {data_dir}: {e}", exc_info=True)
        raise

def load_and_prepare_multimodal_model(models_dir: str, model_weights_path: str, device: torch.device) -> MultimodalModel:
    """
    Loads your MultimodalModel and its state_dict, adjusting keys if necessary.
    """
    logging.info("Attempting to load multimodal model...")

    model_paths = {
    "image": os.path.join(models_dir, "bayesian_model_type:image.pth"),
    "bathy": os.path.join(models_dir, "bayesian_model_type:bathy.pth"),
    "sss": os.path.join(models_dir, "bayesian_model_type:sss.pth"),
    "multimodal": os.path.join(models_dir, "_bayesian_model_type:multimodal.pth")
    }
    num_classes= 7
    const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",
        "moped_enable": True,
        "moped_delta": 0.1,
    }
    models_dict = define_models(model_paths, device=device, num_classes=num_classes, const_bnn_prior_parameters=const_bnn_prior_parameters)
    # Instantiate YOUR actual MultimodalModel class
    multimodal_model = models_dict("multimodal")
    multimodal_model.to(device)

    logging.info(f"Attempting to load state_dict directly into multimodal_model from {model_weights_path}")
    try:
        raw_state_dict = torch.load(model_weights_path, map_location=device)
        logging.debug(f"Raw state dict keys: {raw_state_dict.keys()}") # Use debug for verbose key logging

        new_state_dict = OrderedDict()
        for k, v in raw_state_dict.items():
            # Remove 'module.' prefix (from DataParallel/DDP)
            if k.startswith('module.'):
                k = k[7:]

            # Adjust specific feature extractor prefixes based on your model's saved state_dict
            # If your model's sub-modules are directly named 'image_model_feat', 'sss_model_feat', etc.,
            # and don't have an extra '.model.' in the saved keys, these replacements might not be needed.
            # Adjust these 'if' conditions to match what's in your saved state_dict.
            if k.startswith('image_model_feat.model.'):
                name = k.replace('image_model_feat.model.', 'image_model_feat.', 1)
            elif k.startswith('sss_model_feat.model.'):
                name = k.replace('sss_model_feat.model.', 'sss_model_feat.', 1)
            elif k.startswith('bathy_model_feat.model.'):
                name = k.replace('bathy_model_feat.model.', 'bathy_model_feat.', 1)
            else:
                name = k # Keep as is for other layers (e.g., classifier, shared layers)
            new_state_dict[name] = v

        logging.debug(f"Adjusted state dict keys: {new_state_dict.keys()}") # Use debug for verbose key logging

        # Load the modified state_dict
        # Use strict=True if you expect all keys to match exactly.
        # Use strict=False if you are fine with some keys not matching (e.g., loading a subset of weights).
        load_result = multimodal_model.load_state_dict(new_state_dict, strict=True) # Changed to strict=True for robustness

        missing_keys = load_result.missing_keys
        unexpected_keys = load_result.unexpected_keys

        if missing_keys:
            logging.warning(f"WARNING: The following keys were MISSING in the loaded state_dict compared to the model's state_dict:")
            for key in missing_keys:
                logging.warning(f"  - {key}")
        else:
            logging.info("No missing keys found in the loaded state_dict.")

        if unexpected_keys:
            logging.warning(f"WARNING: The following keys were UNEXPECTED in the loaded state_dict (i.e., present in the loaded weights but not in the model):")
            for key in unexpected_keys:
                logging.warning(f"  - {key}")
        else:
            logging.info("No unexpected keys found in the loaded state_dict.")

        if not missing_keys and not unexpected_keys:
            logging.info("Model state_dict loaded successfully with no missing or unexpected keys.")
        else:
            logging.warning("Model state_dict loaded with some discrepancies. Check warnings above.")

        multimodal_model.train() # Set model to evaluation mode
        return multimodal_model

    except FileNotFoundError:
        logging.error(f"Model weights file not found at: {model_weights_path}")
        raise
    except RuntimeError as re:
        logging.error(f"RuntimeError during state_dict loading (often due to strict=True mismatch): {re}")
        logging.error("Check if your model architecture (MultimodalModel) exactly matches the saved state_dict.")
        logging.error("If you encounter this, try load_state_dict(..., strict=False) to see the mismatches.")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred during model loading or state_dict processing: {e}", exc_info=True)
        raise

# --- Main Execution Block ---
def main(data_directory: str, models_dir:str,  model_weights_file: str, batch_size: int, output_csv: str):
    """
    Main function to run the inference process.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    try:
        # 1. Prepare DataLoader using your CustomImageDataset_1
        inference_dataloader = prepare_inference_dataloader(data_directory, batch_size)

        # 2. Load and Check Multimodal Model using your MultimodalModel
        multimodal_model = load_and_prepare_multimodal_model(models_dir, model_weights_file, device)

        # 3. Perform Inference and Save Results using your multimodal_predict_and_save
        multimodal_predict_and_save(
            multimodal_model=multimodal_model,
            dataloader=inference_dataloader,
            device=device,
            csv_path=output_csv,
            num_mc_samples=1, # Adjust if your actual model uses MC samples (e.g., for BNNs)
            model_type="multimodal"
        )
        logging.info("Final inference process completed successfully.")

    except Exception as e:
        logging.critical(f"A critical error occurred during the inference process: {e}", exc_info=True)
        # Exit with a non-zero status code to indicate failure
        exit(1)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run multimodal AUV inference on a single dataset.")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the directory containing the dataset for inference (e.g., './path/to/my_strangford_data')."
    )
    parser.add_argument(
        "--model_weights",
        type=str,
        required=True,
        help="Path to the PyTorch model weights file (e.g., './multimodal_bnn/pytorch_model.bin')."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for inference. Default: 4."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="./inference_results.csv",
        help="Path to save the inference results CSV. Default: './inference_results.csv'."
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        required=True,
        help="Path to the directory containing the dataset for inference (e.g., './path/to/my_strangford_data')."
    )

    args = parser.parse_args()

    # Call the main function with command-line arguments
    main(
        data_directory=args.data_dir,
        model_dir= args.models_dir,
        model_weights_file=args.model_weights,
        batch_size=args.batch_size,
        output_csv=args.output_csv
    )