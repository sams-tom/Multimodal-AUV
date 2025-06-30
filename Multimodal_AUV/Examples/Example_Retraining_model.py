import os
import logging
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict
import datetime
import sys
from typing import Dict, Any, Tuple # Added Tuple for type hints
from huggingface_hub import hf_hub_download

# Assuming these are your project-specific imports.
# Please ensure MultimodalModel is accessible, e.g., defined in models.model_utils
from Multimodal_AUV.models.model_utils import define_models
# Removed multimodal_predict_and_save as inference is not needed
# Removed CustomImageDataset_1 and prepare_inference_datasets_and_loaders as they are specific to inference or simpler data loading
from Multimodal_AUV.utils.device import move_models_to_device, check_model_devices
from Multimodal_AUV.config.paths import setup_environment_and_devices
from Multimodal_AUV.data.loaders import prepare_datasets_and_loaders # This prepares data for training
from Multimodal_AUV.train.loop_utils import train_and_evaluate_unimodal_model, train_and_evaluate_multimodal_model, define_optimizers_and_schedulers

# TensorBoard is a common utility for training, assuming you use it.
from torch.utils.tensorboard import SummaryWriter

# Set up logging at the module level for clarity
logger = logging.getLogger(__name__)

# --- YOUR PROVIDED load_and_prepare_multimodal_model FUNCTION ---
# This function remains exactly as you provided its logic.
# I've kept the internal rename `load_and_prepare_multimodal_model_custom` for distinctness in this file,
# but its functionality is precisely what you supplied.
def load_and_prepare_multimodal_model_custom( model_weights_path: str, device: torch.device):
    """
    Loads your MultimodalModel and its state_dict, adjusting keys if necessary, exactly as provided.
    """
    logging.info("Attempting to load multimodal model using provided custom loader...")

    num_classes = 7 # Still assuming fixed, adjust if needed
    const_bnn_prior_parameters = {
        "prior_mu": 0.0, "prior_sigma": 1.0, "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0, "type": "Reparameterization",
        "moped_enable": True, "moped_delta": 0.1,
    }
    
    # This calls define_models from your project to get the model instances based on `models_dir`
    models_dict_defined = define_models( device=device[0], num_classes=num_classes, const_bnn_prior_parameters=const_bnn_prior_parameters)
    
    # Correctly access the instantiated MultimodalModel object from the dictionary
    # Assuming "multimodal_model" is the correct key used by your define_models.
    multimodal_model = models_dict_defined["multimodal_model"] 
    multimodal_model.to(device)

    logging.info(f"Attempting to load state_dict directly into multimodal_model from {model_weights_path}")
    try:
        raw_state_dict = torch.load(model_weights_path, map_location=device)
        logging.debug(f"Raw state dict keys: {list(raw_state_dict.keys())}")

        new_state_dict = OrderedDict()
        for k, v in raw_state_dict.items():
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

        logging.debug(f"Adjusted state dict keys: {list(new_state_dict.keys())}")

        load_result = multimodal_model.load_state_dict(new_state_dict, strict=False) 

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

# --- Training Main Function (Streamlined) ---
def training_main(
    const_bnn_prior_parameters: Dict[str, Any],
    multimodal_model_weights_path: str, # Path to your custom saved MultimodalModel weights
    optimizer_params: Dict[str, Dict[str, Any]],
    scheduler_params: Dict[str, Dict[str, Any]],
    training_params: Dict[str, Any],
    root_dir: str, # Now represents the single root for all training data
    devices: list
):
    # Setup logging as per your previous implementation
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "training.log")
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    tb_log_dir = os.path.join("tensorboard_logs", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    sum_writer = SummaryWriter(log_dir=tb_log_dir)
    sum_writer.add_text("Init", "TensorBoard logging started", 0)
    logger.info("Logging initialized.")
    logger.info(f"TensorBoard logs will be saved to: {tb_log_dir}")

    logger.info("Setting up environment and devices...")
    logger.info(f"Using devices: {[str(d) for d in devices]}")

    logger.info("Preparing datasets and data loaders for training...")
    # NEW: Only pass root_dir and batch_size_multimodal
    # Expecting prepare_datasets_and_loaders to return (multimodal_train_loader, multimodal_test_loader, num_classes, _)
    multimodal_train_loader, multimodal_test_loader, num_classes, _ = \
        prepare_datasets_and_loaders(
            root_dir=root_dir,
            batch_size_multimodal=training_params["batch_size_multimodal"]
        )
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Multimodal: {len(multimodal_train_loader.dataset)} training samples, {len(multimodal_test_loader.dataset)} test samples")

    # --- Step 1: Load your custom MultimodalModel with its existing weights using YOUR function ---
    logger.info(f"Loading custom Multimodal Model from {multimodal_model_weights_path} using your provided 'load_and_prepare_multimodal_model_custom' function...")
    try:
        multimodal_model_instance = load_and_prepare_multimodal_model_custom( multimodal_model_weights_path, devices[0])
        logger.info("Custom Multimodal Model loaded successfully with its weights.")
    except Exception as e:
        logger.critical(f"FATAL ERROR: Could not load custom Multimodal Model from {multimodal_model_weights_path}. Cannot proceed with training. Error: {e}")
        sys.exit(1) # Exit if the base model cannot be loaded

    # --- Step 2: Prepare the models_dict for optimizers and device movement ---
    # Only include the multimodal model
    models_dict_for_training = {
        "multimodal_model": multimodal_model_instance
    }

    # Move model to devices (specifically the multimodal_model_instance)
    moved_models_dict = move_models_to_device(models_dict_for_training, devices, use_multigpu_for_multimodal=True)
    multimodal_model_on_device = moved_models_dict["multimodal_model"]
    logger.info("Multimodal model moved to device(s).")
    torch.cuda.empty_cache()

    # --- Step 3: Set up optimizers and schedulers for the loaded model ---
    logger.info("Setting up criterion, optimizers and schedulers...")
    criterion, optimizers, schedulers = define_optimizers_and_schedulers(moved_models_dict, optimizer_params, scheduler_params)

    # --- Step 4: Run Multimodal Training using the loaded model instance ---
    logger.info("Starting multimodal training with the loaded custom model...")
    print("Starting multimodal training...") # Use print for immediate visibility
    train_and_evaluate_multimodal_model(
        train_loader=multimodal_train_loader,
        test_loader=multimodal_test_loader,
        multimodal_model=multimodal_model_on_device, # Pass the exact loaded and moved model instance
        criterion=criterion,
        optimizer=optimizers["multimodal_model"],
        lr_scheduler=schedulers["multimodal_model"],
        num_epochs=training_params["num_epochs_multimodal"],
        device=devices[0], # The primary device for training
        model_type="multimodal",
        bathy_patch_type=training_params["bathy_patch_base"],
        sss_patch_type=training_params["sss_patch_base"],
        csv_path=os.path.join(root_dir, "csvs"), # Assumes csvs dir relative to root_dir
        num_mc=training_params["num_mc"],
        sum_writer=sum_writer
    )
    logger.info("Multimodal training complete.")
    sum_writer.close()
    logger.info("TensorBoard writer closed.")


# --- Orchestrating Main Function ---
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run multimodal AUV model retraining.")
    
    parser.add_argument(
        "--data_dir", # Renamed to root_dir in `training_main` calls, but still `data_dir` for argparse
        type=str,
        required=True,
        help="Path to the root directory containing ALL multimodal dataset for training."
    )
    # REMOVED: --strangford_dir and --mulroy_dir are no longer needed as separate args
    
    # NEW: model_weights argument now represents the Hugging Face repo ID (implicitly)
    # or just signifies that we're using a downloaded model.
    # We will hardcode the repo ID internally for simplicity based on your previous request.
    # If you want this to be configurable, you could change this arg to be '--hf_repo_id'
    # and then use that value in hf_hub_download.
    # For now, we omit it as it's directly derived from fixed HF info.
    
   
    parser.add_argument(
        "--batch_size_multimodal",
        type=int,
        default=4,
        help="Batch size for multimodal training. Default: 4."
    )
    # REMOVED: --batch_size_unimodal
    
    parser.add_argument(
        "--num_epochs_multimodal",
        type=int,
        default=50,
        help="Number of epochs for multimodal training. Default: 50."
    )
    # REMOVED: --num_epochs_unimodal
    
    parser.add_argument(
        "--num_mc_samples",
        type=int,
        default=1,
        help="Number of Monte Carlo samples for BNNs. Default: 1."
    )
    parser.add_argument(
        "--learning_rate_multimodal",
        type=float,
        default=0.001,
        help="Learning rate for multimodal model optimizer. Default: 0.001."
    )
    parser.add_argument(
        "--weight_decay_multimodal",
        type=float,
        default=1e-5,
        help="Weight decay for multimodal model optimizer. Default: 1e-5."
    )
    parser.add_argument(
        "--bathy_patch_base",
        type=str,
        default="none",
        help="Bathy patch type for training. Default: 'none'."
    )
    parser.add_argument(
        "--sss_patch_base",
        type=str,
        default="none",
        help="SSS patch type for training. Default: 'none'."
    )

    args = parser.parse_args()

    # Configure logging at the root level first
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Setup devices based on your utility function
    environment_config = setup_environment_and_devices(root_dir=args.data_dir)
    devices = environment_config["devices"]

    # NEW: Download model weights from Hugging Face Hub
    multimodal_model_hf_repo_id = "sams-tom/multimodal-auv-bathy-bnn-classifier"
    multimodal_model_hf_subfolder = "multimodal-bnn"
    model_weights_filename = os.path.join(multimodal_model_hf_subfolder, "pytorch_model.bin")

    logger.info(f"Attempting to download multimodal model weights from '{multimodal_model_hf_repo_id}/{model_weights_filename}'...")
    try:
        downloaded_model_weights_path = hf_hub_download(
            repo_id=multimodal_model_hf_repo_id,
            filename=model_weights_filename
        )
        logger.info(f"Multimodal model weights downloaded to: {downloaded_model_weights_path}")
    except Exception as e:
        logger.critical(f"FATAL ERROR: Could not download model weights from Hugging Face Hub. Error: {e}")
        sys.exit(1)

    # Common parameters for model definition
    const_bnn_prior_parameters = {
        "prior_mu": 0.0, "prior_sigma": 1.0, "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0, "type": "Reparameterization",
        "moped_enable": True, "moped_delta": 0.1,
    }
   

    # Define optimizer and scheduler parameters (only multimodal_model needed)
    optimizer_params = {
        "multimodal_model": {"type": "Adam", "lr": args.learning_rate_multimodal, "weight_decay": args.weight_decay_multimodal},
    }

    scheduler_params = {
        "multimodal_model": {"type": "ReduceLROnPlateau", "mode": "min", "factor": 0.5, "patience": 5},
    }

    # Define training parameters (only multimodal-specific)
    training_params = {
        "batch_size_multimodal": args.batch_size_multimodal,
        "num_epochs_multimodal": args.num_epochs_multimodal,
        "num_mc": args.num_mc_samples,
        "bathy_patch_base": args.bathy_patch_base,
        "sss_patch_base": args.sss_patch_base,
    }

    # Directly call training_main
    training_main(
        const_bnn_prior_parameters=const_bnn_prior_parameters,
        multimodal_model_weights_path=downloaded_model_weights_path, # Pass the downloaded path
        optimizer_params=optimizer_params,
        scheduler_params=scheduler_params,
        training_params=training_params,
        root_dir=args.data_dir, # This is now the single root for all multimodal training data
        devices=devices
    )

if __name__ == "__main__":
    main()