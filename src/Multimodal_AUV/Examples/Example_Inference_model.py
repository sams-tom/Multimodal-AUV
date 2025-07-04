import os
import logging
import torch
from torch.utils.data import DataLoader, ConcatDataset
from collections import OrderedDict
import pandas as pd
from huggingface_hub import hf_hub_download # NEW: Import for Hugging Face Hub download
import argparse

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
    #Attempts to load the data directory
    try:
        if not os.path.exists(data_dir):
            logging.error(f"Dataset directory not found: {data_dir}")
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

        #Organises this into a Custom dataset (containing sonar and image)
        dataset = CustomImageDataset_1(data_dir)
        if len(dataset) == 0:
            logging.error(f"No samples found in CustomImageDataset_1 from {data_dir}. Check your dataset implementation and directory contents.")
            raise ValueError(f"No samples found in dataset from {data_dir}")

        #Turn this dataset into a dataloader and return this
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        logging.info(f"DataLoader prepared for '{data_dir}' with {len(dataset)} items.")
        return dataloader
    except Exception as e:
        logging.error(f"Error preparing inference DataLoader from {data_dir}: {e}", exc_info=True)
        raise

def load_and_prepare_multimodal_model(downloaded_model_weights_path: str, device: torch.device, num_classes: int) -> torch.nn.Module:
    """
    Loads your MultimodalModel and its state_dict, adjusting keys if necessary.
    Takes the already downloaded model_weights_path.
    """
    logging.info("Attempting to load multimodal model...")

   #These are set to load the model
    const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",
        "moped_enable": True,
        "moped_delta": 0.1,
    }

    #Calls the model defining function
    models_dict_instances = define_models(device=device, num_classes=num_classes, const_bnn_prior_parameters=const_bnn_prior_parameters)


    #This checks the multimodal model has been defined
    if "multimodal_model" not in models_dict_instances:
        logging.error("Key 'multimodal_model' not found in models_dict returned by define_models. Check define_models implementation.")
        raise KeyError("Multimodal model instance not found in define_models output.")

    #Extracts this multimodal model and moves it to device
    multimodal_model = models_dict_instances["multimodal_model"]
    multimodal_model.to(device)

    logging.info(f"Attempting to load state_dict directly into multimodal_model from {downloaded_model_weights_path}")

    try:
        #Loads the model
        raw_state_dict = torch.load(downloaded_model_weights_path, map_location=device)
        logging.debug(f"Raw state dict keys: {raw_state_dict.keys()}")

        #This below cleans up the loading file so that the downloaded model and the defined model line up
        new_state_dict = OrderedDict()
        for k, v in raw_state_dict.items():
            # Remove 'module.' prefix (from DataParallel/DDP)
            if k.startswith('module.'):
                k = k[7:]

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
        if num_classes != 7:
            logging.warning(f"WARNING: The model was trained with 7 classes, but num_classes is set to {num_classes}. Therefore the final output layer is dropped for retraining.")
            keys_to_remove = []
            for key in new_state_dict.keys():
                if key.startswith('fc2.'):
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del new_state_dict[key]
                logging.info(f"Removed key '{key}' from loaded state_dict due to expected size mismatch.")

        #Next it loads this model
        ##NOTE: THEY DONT PERFECTLY LINE UP BUT THIS IS FINE
        load_result = multimodal_model.load_state_dict(new_state_dict, strict=False)

        missing_keys = load_result.missing_keys
        unexpected_keys = load_result.unexpected_keys

        #This defines a collection of warnings
        if missing_keys:
            logging.warning(f"WARNING:  ONLY WARNING, THIS IS NOT THE END OF THE WORLD IF LIST IS APPROX 9 FC LAYERS, THIS IS EXPECTED. The following keys were MISSING in the loaded state_dict compared to the model's state_dict:")
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

        #sets model to train to allow the uncertainty calculation
        multimodal_model.train() 
       
        #Returns the mutlimodal model
        return multimodal_model

    #Error handling
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
def main(data_directory: str, batch_size: int, output_csv: str, num_mc_samples: int, num_classes: int): # NEW: Added num_classes
    """
    Main function to run the inference process.
    """

    #Uses just one gpu if avalible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    try:
        ##The below loads the mutlimodal model weights from hugging face

        #Defining the folder structure on hugging face
        multimodal_model_hf_repo_id = "sams-tom/multimodal-auv-bathy-bnn-classifier"
        multimodal_model_hf_subfolder = "multimodal-bnn"
        model_weights_filename = os.path.join(multimodal_model_hf_subfolder, "pytorch_model.bin")

        logging.info(f"Attempting to download multimodal model weights from '{multimodal_model_hf_repo_id}/{model_weights_filename}'...")
        #Downloads the weights
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
        # Pass the path to the downloaded weights and num_classes
        multimodal_model = load_and_prepare_multimodal_model(downloaded_model_weights_path, device, num_classes) 


        with torch.no_grad():
            # 3. Perform Inference and Save Results using your multimodal_predict_and_save
            multimodal_predict_and_save(
                multimodal_model=multimodal_model,
                dataloader=inference_dataloader,
                device=device,
                csv_path=output_csv,
                num_mc_samples=num_mc_samples, 
                model_type="multimodal"
            )
        logging.info("Final inference process completed successfully.")

    except Exception as e:
        logging.critical(f"A critical error occurred during the inference process: {e}", exc_info=True)
        # Exit with a non-zero status code to indicate failure
        exit(1)

if __name__ == "__main__":

    #Define the aug paser
    parser = argparse.ArgumentParser(description="Run multimodal AUV inference on a single dataset.")

    #Requires a data directory
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the directory containing the dataset for inference (e.g., './path/to/my_strangford_data')."
    )
    #Requires a batch size
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for inference. Default: 4."
    )
    #Requires a path for the output csv
    parser.add_argument(
        "--output_csv",
        type=str,
        default="./inference_results.csv",
        help="Path to save the inference results CSV. Default: './inference_results.csv'."
    )
    #Requires a number of multicarlo samples for uncertainty quantification
    parser.add_argument( 
        "--num_mc_samples",
        type=int,
        default=5,
        help="Number of Monte Carlo samples to draw for BNN inference. Default: 1 (no MC dropout)."
    )
    #Requires a number of classes for model instatiation.
    #Note this must equal 7 for the downloaded model to work
    parser.add_argument( 
        "--num_classes",
        type=int,
        required=True, 
        default=7,
        help="Number of output classes for the classification model. Note for the downloaded model this must be 7"
    )

    args = parser.parse_args()

    # Call the main function with command-line arguments
    main(
        data_directory=args.data_dir,
        batch_size=args.batch_size,
        output_csv=args.output_csv,
        num_mc_samples=args.num_mc_samples,
        num_classes=args.num_classes 
    )
