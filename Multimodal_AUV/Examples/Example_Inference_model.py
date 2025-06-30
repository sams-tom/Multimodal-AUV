import os
import logging
import torch
from torch.utils.data import DataLoader, ConcatDataset
from collections import OrderedDict
import pandas as pd
from huggingface_hub import hf_hub_download # NEW: Import for Hugging Face Hub download

# Assuming these are correctly defined and accessible in your project
from Multimodal_AUV.models.model_utils import define_models
from Multimodal_AUV.inference.predictors import multimodal_predict_and_save
from Multimodal_AUV.data.datasets import CustomImageDataset_1

# Configure logging for better output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def load_and_prepare_multimodal_model( downloaded_model_weights_path: str, device: torch.device) -> torch.nn.Module:
    """
    Loads your MultimodalModel and its state_dict, adjusting keys if necessary.
    Takes the already downloaded model_weights_path.
    """
    logging.info("Attempting to load multimodal model...")

   
    num_classes = 7 # Assuming fixed for this model, or pass as arg if variable
    const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",
        "moped_enable": True,
        "moped_delta": 0.1,
    }

    # IMPORTANT: The `models_dict = define_models(...)` call is crucial.
    # It seems to define individual Bayesian models and also your 'multimodal' model.
    # Ensure that `define_models` correctly sets up the multimodal architecture
    # that matches the `pytorch_model.bin` weights you're downloading.
    models_dict_instances = define_models(device=device[0], num_classes=num_classes, const_bnn_prior_parameters=const_bnn_prior_parameters)

    # Instantiate YOUR actual MultimodalModel class from the returned dictionary
    # Assuming define_models returns a dictionary where "multimodal" key gives the instantiated model
    if "multimodal_model" not in models_dict_instances:
        logging.error("Key 'multimodal_model' not found in models_dict returned by define_models. Check define_models implementation.")
        raise KeyError("Multimodal model instance not found in define_models output.")

    multimodal_model = models_dict_instances["multimodal_model"]
    multimodal_model.to(device)

    logging.info(f"Attempting to load state_dict directly into multimodal_model from {downloaded_model_weights_path}")
    try:
        raw_state_dict = torch.load(downloaded_model_weights_path, map_location=device)
        logging.debug(f"Raw state dict keys: {raw_state_dict.keys()}")

        new_state_dict = OrderedDict()
        for k, v in raw_state_dict.items():
            # Remove 'module.' prefix (from DataParallel/DDP)
            if k.startswith('module.'):
                k = k[7:]

            # Adjust specific feature extractor prefixes based on your model's saved state_dict
            # These adjustments are critical for `load_state_dict` to work.
            # Make sure these prefixes match *exactly* what's in your saved `pytorch_model.bin`
            # and what your `MultimodalModel` expects.
            if k.startswith('image_model_feat.model.'):
                name = k.replace('image_model_feat.model.', 'image_model_feat.', 1)
            elif k.startswith('sss_model_feat.model.'):
                name = k.replace('sss_model_feat.model.', 'sss_model_feat.', 1)
            elif k.startswith('bathy_model_feat.model.'):
                name = k.replace('bathy_model_feat.model.', 'bathy_model_feat.', 1)
            else:
                name = k
            new_state_dict[name] = v

        logging.debug(f"Adjusted state dict keys: {new_state_dict.keys()}")

        load_result = multimodal_model.load_state_dict(new_state_dict, strict=True)

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

        multimodal_model.train() # Set model to evaluation mode; if inference, should be .eval()
       
        return multimodal_model

    except FileNotFoundError:
        logging.error(f"Model weights file not found at: {downloaded_model_weights_path}")
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
def main(data_directory: str, models_dir: str, batch_size: int, output_csv: str, num_mc_samples: int): # NEW: Added num_mc_samples
    """
    Main function to run the inference process.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    try:
        # --- NEW: Download model weights from Hugging Face Hub ---
        multimodal_model_hf_repo_id = "sams-tom/multimodal-auv-bathy-bnn-classifier"
        multimodal_model_hf_subfolder = "multimodal-bnn"
        model_weights_filename = os.path.join(multimodal_model_hf_subfolder, "pytorch_model.bin")

        logging.info(f"Attempting to download multimodal model weights from '{multimodal_model_hf_repo_id}/{model_weights_filename}'...")
        downloaded_model_weights_path = hf_hub_download(
            repo_id=multimodal_model_hf_repo_id,
            filename=model_weights_filename,
            # If you want to specify a cache directory: cache_dir="./hf_cache"
        )
        logging.info(f"Multimodal model weights downloaded to: {downloaded_model_weights_path}")
        # --- END NEW DOWNLOAD BLOCK ---

        # 1. Prepare DataLoader using your CustomImageDataset_1
        inference_dataloader = prepare_inference_dataloader(data_directory, batch_size)

        # 2. Load and Check Multimodal Model using your MultimodalModel
        # Pass the path to the downloaded weights
        multimodal_model = load_and_prepare_multimodal_model(models_dir, downloaded_model_weights_path, device) # NEW: Pass downloaded path

        # Set model to evaluation mode and disable gradient calculations for inference
        multimodal_model.eval()
        with torch.no_grad():
            # 3. Perform Inference and Save Results using your multimodal_predict_and_save
            multimodal_predict_and_save(
                multimodal_model=multimodal_model,
                dataloader=inference_dataloader,
                device=device,
                csv_path=output_csv,
                num_mc_samples=num_mc_samples, # NEW: Pass num_mc_samples from argument
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
    # REMOVED: --model_weights is now handled by internal download
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

    parser.add_argument( # NEW: Add num_mc_samples argument
        "--num_mc_samples",
        type=int,
        default=1,
        help="Number of Monte Carlo samples to draw for BNN inference. Default: 1 (no MC dropout)."
    )

    args = parser.parse_args()

    # Call the main function with command-line arguments
    main(
        data_directory=args.data_dir,
        batch_size=args.batch_size,
        output_csv=args.output_csv,
        num_mc_samples=args.num_mc_samples # NEW: Pass num_mc_samples
    )