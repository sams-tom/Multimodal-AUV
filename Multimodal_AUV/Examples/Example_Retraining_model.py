import os
import logging
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict
import datetime
import sys
from typing import Dict, Any

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
def load_and_prepare_multimodal_model_custom(models_dir: str, model_weights_path: str, device: torch.device):
    """
    Loads your MultimodalModel and its state_dict, adjusting keys if necessary, exactly as provided.
    """
    logging.info("Attempting to load multimodal model using provided custom loader...")

    model_paths = {
        "image": os.path.join(models_dir, "bayesian_model_type:image.pth"),
        "bathy": os.path.join(models_dir, "bayesian_model_type:bathy.pth"),
        "sss": os.path.join(models_dir, "bayesian_model_type:sss.pth"),
        "multimodal": os.path.join(models_dir, "_bayesian_model_type:multimodal.pth")
    }
    num_classes = 7
    const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",
        "moped_enable": True,
        "moped_delta": 0.1,
    }
    
    # This calls define_models from your project to get the model instances
    models_dict_defined = define_models(model_paths, device=device, num_classes=num_classes, const_bnn_prior_parameters=const_bnn_prior_parameters)
    
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
    model_paths: Dict[str, str], # Paths used by define_models to create the structure
    multimodal_model_weights_path: str, # Path to your custom saved MultimodalModel weights
    optimizer_params: Dict[str, Dict[str, Any]],
    scheduler_params: Dict[str, Dict[str, Any]],
    training_params: Dict[str, Any],
    root_dir: str,
    models_dir: str,
    strangford_dir: str,
    mulroy_dir: str,
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
    unimodal_train_loader, unimodal_test_loader, multimodal_train_loader, multimodal_test_loader, num_classes, _ = \
        prepare_datasets_and_loaders(
            root_dir=root_dir,
            batch_size_unimodal=training_params["batch_size_unimodal"],
            batch_size_multimodal=training_params["batch_size_multimodal"],
            strangford_dir=strangford_dir,
            mulroy_dir=mulroy_dir
        )
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Unimodal: {len(unimodal_train_loader.dataset)} training samples, {len(unimodal_test_loader.dataset)} test samples")
    logger.info(f"Multimodal: {len(multimodal_train_loader.dataset)} training samples, {len(multimodal_test_loader.dataset)} test samples")

    # --- Step 1: Load your custom MultimodalModel with its existing weights using YOUR function ---
    logger.info(f"Loading custom Multimodal Model from {multimodal_model_weights_path} using your provided 'load_and_prepare_multimodal_model_custom' function...")
    try:
        multimodal_model_instance = load_and_prepare_multimodal_model_custom(models_dir, multimodal_model_weights_path, devices[0])
        logger.info("Custom Multimodal Model loaded successfully with its weights.")
    except Exception as e:
        logger.critical(f"FATAL ERROR: Could not load custom Multimodal Model from {multimodal_model_weights_path}. Cannot proceed with training. Error: {e}")
        sys.exit(1) # Exit if the base model cannot be loaded

    # --- Step 2: Prepare the models_dict for optimizers and device movement ---
    # We now have the `multimodal_model_instance` ready.
    # Create a dictionary suitable for `define_optimizers_and_schedulers` and `move_models_to_device`.
    models_dict_for_training = {
        "multimodal_model": multimodal_model_instance
        # If your training also includes separate unimodal models (e.g., if you call train_and_evaluate_unimodal_model),
        # they would need to be instantiated here (e.g., via define_models) and added to this dictionary.
        # For this request, we assume focus is solely on the multimodal model.
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
    print("Starting multimodal training...")
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
        csv_path=os.path.join(root_dir, "csvs"),
        num_mc=training_params["num_mc"],
        sum_writer=sum_writer
    )
    logger.info("Multimodal training complete.")
    sum_writer.close()
    logger.info("TensorBoard writer closed.")


# --- Orchestrating Main Function (Now ONLY for Training) ---
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run multimodal AUV model retraining.")
    
    # Removed --mode as it's only training
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True, # Made required as it's root_dir for training data loaders
        help="Path to the root directory containing the dataset for training."
    )
    parser.add_argument(
        "--strangford_dir",
        type=str,
        required=True, # Made required as it's needed for prepare_datasets_and_loaders
        help="Path to the Strangford dataset directory for training."
    )
    parser.add_argument(
        "--mulroy_dir",
        type=str,
        required=True, # Made required as it's needed for prepare_datasets_and_loaders
        help="Path to the Mulroy dataset directory for training."
    )
    parser.add_argument(
        "--model_weights",
        type=str,
        required=True, # Still required for loading your base model weights
        help="Path to the PyTorch model weights file for loading your custom MultimodalModel for retraining."
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        required=True,
        help="Path to the directory containing the model definition files (e.g., './Multimodal_AUV/models'). This is used by define_models."
    )
    parser.add_argument(
        "--batch_size_multimodal", # Renamed from --batch_size for clarity as it's only multimodal training now
        type=int,
        default=4,
        help="Batch size for multimodal training. Default: 4."
    )
    parser.add_argument(
        "--batch_size_unimodal",
        type=int,
        default=8,
        help="Batch size for unimodal training (if enabled in prepare_datasets_and_loaders, though not directly used by current training loop). Default: 8."
    )
    # Removed --output_csv_inference as inference is gone
    parser.add_argument(
        "--num_epochs_multimodal",
        type=int,
        default=50,
        help="Number of epochs for multimodal training. Default: 50."
    )
    parser.add_argument(
        "--num_epochs_unimodal", # Kept in case you eventually train unimodal separately, but not directly used by this script's `training_main`
        type=int,
        default=10,
        help="Number of epochs for unimodal training (if enabled). Default: 10."
    )
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
    # Use args.data_dir directly as it's required for training.
    environment_config = setup_environment_and_devices(root_dir=args.data_dir)
    devices = environment_config["devices"]

    # Common parameters for model definition
    const_bnn_prior_parameters = {
        "prior_mu": 0.0, "prior_sigma": 1.0, "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0, "type": "Reparameterization",
        "moped_enable": True, "moped_delta": 0.1,
    }
    # These model_paths are used internally by define_models, which is called by
    # your load_and_prepare_multimodal_model_custom function to build the model *structure*.
    model_paths = {
        "image": os.path.join(args.models_dir, "bayesian_model_type:image.pth"),
        "bathy": os.path.join(args.models_dir, "bayesian_model_type:bathy.pth"),
        "sss": os.path.join(args.models_dir, "bayesian_model_type:sss.pth"),
        "multimodal": os.path.join(args.models_dir, "_bayesian_model_type:multimodal.pth")
    }

    # Define optimizer and scheduler parameters
    optimizer_params = {
        # Only "multimodal_model" is strictly needed for the current `training_main` loop.
        # Others kept for completeness if you have unimodal training in mind for future use.
        "image_model": {"type": "Adam", "lr": 0.001, "weight_decay": 1e-5},
        "bathy_model": {"type": "Adam", "lr": 0.001, "weight_decay": 1e-5},
        "sss_model": {"type": "Adam", "lr": 0.001, "weight_decay": 1e-5},
        "multimodal_model": {"type": "Adam", "lr": args.learning_rate_multimodal, "weight_decay": args.weight_decay_multimodal},
    }

    scheduler_params = {
        # Only "multimodal_model" is strictly needed.
        "image_model": {"type": "ReduceLROnPlateau", "mode": "min", "factor": 0.5, "patience": 5},
        "bathy_model": {"type": "ReduceLROnPlateau", "mode": "min", "factor": 0.5, "patience": 5},
        "sss_model": {"type": "ReduceLROnPlateau", "mode": "min", "factor": 0.5, "patience": 5},
        "multimodal_model": {"type": "ReduceLROnPlateau", "mode": "min", "factor": 0.5, "patience": 5},
    }

    # Define training parameters
    training_params = {
        "batch_size_unimodal": args.batch_size_unimodal,
        "batch_size_multimodal": args.batch_size_multimodal,
        "num_epochs_unimodal": args.num_epochs_unimodal,
        "num_epochs_multimodal": args.num_epochs_multimodal,
        "num_mc": args.num_mc_samples,
        "bathy_patch_base": args.bathy_patch_base,
        "sss_patch_base": args.sss_patch_base,
    }

    # Directly call training_main as there's no other mode
    training_main(
        const_bnn_prior_parameters=const_bnn_prior_parameters,
        model_paths=model_paths,
        multimodal_model_weights_path=args.model_weights,
        optimizer_params=optimizer_params,
        scheduler_params=scheduler_params,
        training_params=training_params,
        root_dir=args.data_dir,
        models_dir=args.models_dir,
        strangford_dir=args.strangford_dir,
        mulroy_dir=args.mulroy_dir,
        devices=devices
    )

if __name__ == "__main__":
    main()