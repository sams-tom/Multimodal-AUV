import os
import logging
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict
import datetime
import sys
from typing import Dict, Any, Tuple 
from huggingface_hub import hf_hub_download
from torch.utils.tensorboard import SummaryWriter
import argparse


#Load in project specific imports
from Multimodal_AUV.models.model_utils import define_models
from Multimodal_AUV.utils.device import move_models_to_device, check_model_devices
from Multimodal_AUV.config.paths import get_empty_gpus
from Multimodal_AUV.data.loaders import prepare_datasets_and_loaders
from Multimodal_AUV.train.loop_utils import train_and_evaluate_unimodal_model, train_and_evaluate_multimodal_model, define_optimizers_and_schedulers


# Set up logging at the module level for clarity
logger = logging.getLogger(__name__)


def load_and_prepare_multimodal_model_custom(model_weights_path: str, device: torch.device, num_classes: int) -> torch.nn.Module:
    """
    Loads your MultimodalModel and its state_dict, adjusting keys if necessary, exactly as provided.
    """
    logging.info("Attempting to load multimodal model using provided custom loader...")

    #These are defined to load the model
    const_bnn_prior_parameters = {
        "prior_mu": 0.0, "prior_sigma": 1.0, "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0, "type": "Reparameterization",
        "moped_enable": True, "moped_delta": 0.1,
    }
    
    #Call the model defintiiosn
    models_dict_defined = define_models( device=device, num_classes=num_classes, const_bnn_prior_parameters=const_bnn_prior_parameters)
    
    # Correctly access the instantiated MultimodalModel object from the dictionary and move to device
    multimodal_model = models_dict_defined["multimodal_model"] 
    multimodal_model.to(device)

    logging.info(f"Attempting to load state_dict directly into multimodal_model from {model_weights_path}")


    try:
        #Load the model
        raw_state_dict = torch.load(model_weights_path, map_location=device)
        logging.debug(f"Raw state dict keys: {list(raw_state_dict.keys())}")

        #This is to clean up and harmonise the defined model and the downloaded model
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

        #Final load the downloaded model
        load_result = multimodal_model.load_state_dict(new_state_dict, strict=False) 

        #Find missing keys and report these as warnings
        missing_keys = load_result.missing_keys
        unexpected_keys = load_result.unexpected_keys

        if missing_keys:
            logging.warning(f"WARNING. ONLY A WANRING IF ~9 FC LAYERS THIS IS EXPECTED: The following keys were MISSING in the loaded state_dict compared to the model's state_dict:")
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

    #Error definition
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

def training_main(
    multimodal_model_weights_path: str,
    optimizer_params: Dict[str, Dict[str, Any]],
    scheduler_params: Dict[str, Dict[str, Any]],
    training_params: Dict[str, Any],
    root_dir: str,
    devices: list,
    const_bnn_prior_parameters: Dict[str, Any],
        num_classes: int 

):
    # Setup logging and tensor board
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

    #Some preliminary information
    logger.info("Logging initialized.")
    logger.info(f"TensorBoard logs will be saved to: {tb_log_dir}")
    logger.info("Setting up environment and devices...")
    logger.info(f"Using devices: {[str(d) for d in devices]}")

    logger.info("Preparing datasets and data loaders for training...")

    #Define the multimodal loaders and get the number of classes. Note _ are unimodal and not needed 
    _, _, multimodal_train_loader, multimodal_test_loader, num_classes, _ = \
        prepare_datasets_and_loaders(
            root_dir=root_dir,
            batch_size_multimodal=training_params["batch_size_multimodal"],
            batch_size_unimodal=1 
        )

    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Multimodal: {len(multimodal_train_loader.dataset)} training samples, {len(multimodal_test_loader.dataset)} test samples")

    # --- Step 1: Load your custom MultimodalModel with its existing weights using YOUR function ---
    logger.info(f"Loading custom Multimodal Model from {multimodal_model_weights_path} using your provided 'load_and_prepare_multimodal_model_custom' function...")
    try:
       #Load the multimodal model form hugging face
        multimodal_model_instance = load_and_prepare_multimodal_model_custom(
            multimodal_model_weights_path,
            devices[0], num_classes=num_classes
      
        )
        logger.info("Custom Multimodal Model loaded successfully with its weights.")
    except Exception as e:
        logger.critical(f"FATAL ERROR: Could not load custom Multimodal Model from {multimodal_model_weights_path}. Cannot proceed with training. Error: {e}")
        sys.exit(1) # Exit if the base model cannot be loaded

    # 2. Define optimiser and schedulers
    #Defining the models so the optimisers know what to do
    logging.info("Defining models...")
    models_dict = define_models(device=devices[0], num_classes=num_classes, const_bnn_prior_parameters=const_bnn_prior_parameters)
    models_dict = move_models_to_device(models_dict, devices, use_multigpu_for_multimodal=True)
    logging.info("Models moved to devices.")
    torch.cuda.empty_cache()

    #Call the optimisers and scheudlers
    logging.info("Setting up criterion, optimizers and schedulers...")
    criterion, optimizers, schedulers = define_optimizers_and_schedulers(models_dict, optimizer_params, scheduler_params)

    #3. Call the main loop with the saved model to retrain
    logger.info("Starting multimodal training with the loaded custom model...")
    print("Starting multimodal training...") # Use print for immediate visibility
    train_and_evaluate_multimodal_model(
        train_loader=multimodal_train_loader,
        test_loader=multimodal_test_loader,
        multimodal_model=multimodal_model_instance, 
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

    #Define the paser
    parser = argparse.ArgumentParser(description="Run multimodal AUV model retraining.")

    #Requires a data directory
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the root directory containing ALL multimodal dataset for training."
    )

    #Requires a batch size
    parser.add_argument(
        "--batch_size_multimodal",
        type=int,
        default=4,
        help="Batch size for multimodal training. Default: 4."
    )

    #Requires a number of epochs for retraining
    parser.add_argument(
        "--num_epochs_multimodal",
        type=int,
        default=50,
        help="Number of epochs for multimodal training. Default: 50."
    )

    #Requires a number of monte carlo samples for uncertainty quantification
    parser.add_argument(
        "--num_mc_samples",
        type=int,
        default=5,
        help="Number of Monte Carlo samples for BNNs. Default: 1."
    )

    #Requires a learning rate, this was optimised and found to be optimal
    parser.add_argument(
        "--learning_rate_multimodal",
        type=float,
        default= 5e-5,
        help="Learning rate for multimodal model optimizer. Note this default was optimised and found to be optimal Default:  5e-5."
    )

    #Requires a weight decay
    parser.add_argument(
        "--weight_decay_multimodal",
        type=float,
        default=1e-5,
        help="Weight decay for multimodal model optimizer. Note this default was optimised and found to be optimal Default: 1e-5."
    )

    #Set the bathymetric sonar patch size if multiple in data directory
    parser.add_argument(
        "--bathy_patch_base",
        type=str,
        default="none",
        help="Bathy patch type for training. Default: 'none'."
    )
    #Set the side scan path size if multiple in data directory
    parser.add_argument(
        "--sss_patch_base",
        type=str,
        default="none",
        help="SSS patch type for training. Default: 'none'."
    )
    #Set number of classes . NOTE this must be 7 to download model
    parser.add_argument(
        "--num_classes",
        type=int,
        default=7,
        help="Number of classes for model deifnition, must be 7 if downloading model. Default: '7'."
    )

    args = parser.parse_args()

    # Configure logging at the root level first
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Setup devices finding number of free GPUs or setting as CPU
    devices = []
    if torch.cuda.is_available():
        devices = get_empty_gpus(threshold_mb=1000) # You can adjust this threshold
        if not devices:
            print("No empty GPUs found or CUDA not available. Falling back to CPU.")
            devices = [torch.device("cpu")]
        else:
            print(f"Selected empty GPUs: {[str(d) for d in devices]}")
    else:
        devices = [torch.device("cpu")]
        print("CUDA not available. Using CPU.")


    # Download model weights from Hugging Face Hub
    #Defining folders
    multimodal_model_hf_repo_id = "sams-tom/multimodal-auv-bathy-bnn-classifier"
    multimodal_model_hf_subfolder = "multimodal-bnn"
    model_weights_filename = os.path.join(multimodal_model_hf_subfolder, "pytorch_model.bin")

    logger.info(f"Attempting to download multimodal model weights from '{multimodal_model_hf_repo_id}/{model_weights_filename}'...")
    try:
        #Downloading model from hugging face
        downloaded_model_weights_path = hf_hub_download(
            repo_id=multimodal_model_hf_repo_id,
            filename=model_weights_filename
        )
        logger.info(f"Multimodal model weights downloaded to: {downloaded_model_weights_path}")
    except Exception as e:
        logger.critical(f"FATAL ERROR: Could not download model weights from Hugging Face Hub. Error: {e}")
        sys.exit(1)

    # Define optimizer and scheduler parameters (only multimodal_model needed)
    optimizer_params = {
        "image_model": {"lr": 1e-5},
        "bathy_model": {"lr": 0.01},
        "sss_model": {"lr": 1e-5},
        "multimodal_model": {"lr": 5e-5}
    }

    scheduler_params = {
        "image_model": {"step_size": 7, "gamma": 0.1},
        "bathy_model": {"step_size": 5, "gamma": 0.5},
        "sss_model": {"step_size": 7, "gamma": 0.7},
        "multimodal_model": {"step_size": 7, "gamma": 0.752}
    }

    # Define training parameters (only multimodal-specific)
    training_params = {
        "batch_size_multimodal": args.batch_size_multimodal,
        "num_epochs_multimodal": args.num_epochs_multimodal,
        "num_mc": args.num_mc_samples,
        "bathy_patch_base": args.bathy_patch_base,
        "sss_patch_base": args.sss_patch_base,
    }

    # --- Define Bayesian parameters statically within main() ---
    const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",
        "moped_enable": True,
        "moped_delta": 0.1,
    }

    # Directly call training_main
    training_main(
        multimodal_model_weights_path=downloaded_model_weights_path,
        optimizer_params=optimizer_params,
        scheduler_params=scheduler_params,
        training_params=training_params,
        root_dir=args.data_dir,
        devices=devices,
        const_bnn_prior_parameters=const_bnn_prior_parameters # Pass the statically defined BNN constants
    )

if __name__ == "__main__":
    main()