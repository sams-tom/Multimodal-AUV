# src/multimodal_auv/cli.py
import torch 
import os
import argparse
import logging
import sys
# If you need to load YAML config files within these CLIs, ensure pyyaml is imported
import yaml 

# Configure basic logging for CLI
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- IMPORTANT: Import your core functions exposed by your __init__.py ---
# Make sure your __init__.py actually imports these from 'functions/functions.py'
# as per your example: from .functions.functions import ...
from Multimodal_AUV.functions.functions import (
    run_auv_inference,
    run_auv_training,
    run_auv_preprocessing,
    run_AUV_training_from_scratch
)

# --- CLI Function for Data Preparation (from Example_data_preparation.py) ---
def data_preparation_cli():
    parser = argparse.ArgumentParser(
        description="Preprocess AUV sonar and optical image data for machine learning tasks."
    )
    # Reconstructing arguments from your example command
    parser.add_argument("--raw_optical_images_folder", type=str, required=True,
                        help="Path to the folder containing raw optical image files.")
    parser.add_argument("--geotiff_folder", type=str, required=True,
                        help="Path to the folder containing AUV bathymetry GeoTIFF files.")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Path to the output folder for processed data.")
    # Note: For Linux, 'C:/exiftool/' is a Windows path. Ensure this is correct for Linux.
    parser.add_argument("--exiftool_path", type=str, required=True,
                        help="Path to the ExifTool executable (e.g., '/usr/bin/exiftool' on Linux).")
    parser.add_argument("--window_size_meters", type=float, default=20.0, # Default based on previous discussion
                        help="Window size in meters for neighborhood aggregation (default: 20.0).")
    parser.add_argument("--image_enhancement_method", type=str, default="AverageSubtraction",
                        choices=["AverageSubtraction", "CLAHE", "None"],
                        help="Image enhancement method for optical images (default: AverageSubtraction).")
    # If run_auv_preprocessing supports skip_bathy_combine, add it here:
    # parser.add_argument("--skip_bathy_combine", action="store_true",
    #                     help="If set, skip the bathymetry combination step (useful for re-runs).")

    args = parser.parse_args()
    logging.info("Starting AUV data preparation pipeline...")
    try:
        # Call your core function: run_auv_preprocessing
        # Adjust arguments based on what run_auv_preprocessing actually accepts
        success = run_auv_preprocessing(
            raw_optical_images_folder=args.raw_optical_images_folder,
            geotiff_folder=args.geotiff_folder,
            output_folder=args.output_folder,
            exiftool_path=args.exiftool_path,
            window_size_meters=args.window_size_meters,
            image_enhancement_method=args.image_enhancement_method
            # ... pass other args ...
        )
        if not success:
            logging.error("Data preparation pipeline failed.")
            sys.exit(1)
        logging.info("Data preparation pipeline completed successfully!")
    except Exception as e:
        logging.exception(f"An unexpected error occurred during data preparation: {e}")
        sys.exit(1)

# --- CLI Function for Inference (from Example_Inference_model.py) ---
def inference_cli():
    parser = argparse.ArgumentParser(
        description="Run inference using a trained Multimodal AUV model."
    )
    # Reconstructing arguments from your example command
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the input data directory for inference.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the inference results CSV.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference (default: 4).")
    parser.add_argument("--num_mc_samples", type=int, default=20, help="Number of Monte Carlo samples for uncertainty (default: 20).")
    # Add any other arguments run_auv_inference accepts

    args = parser.parse_args()
    logging.info(f"Starting AUV inference with data from: {args.data_dir}")
    try:
        # Call your core function: run_auv_inference
        success = run_auv_inference(
            data_directory=args.data_dir,
            output_csv=args.output_csv,
            batch_size=args.batch_size,
            num_mc_samples=args.num_mc_samples
            # ... pass other args ...
        )
        if not success:
            logging.error("Inference failed.")
            sys.exit(1)
        logging.info("Inference completed successfully!")
    except Exception as e:
        logging.exception(f"An unexpected error occurred during inference: {e}")
        sys.exit(1)

# --- CLI Function for Retraining Model (from Example_Retraining_model.py) ---
def retraining_cli():
    parser = argparse.ArgumentParser(
        description="Retrain a Multimodal AUV model."
    )
    # Reconstructing arguments from your example command
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the training data directory.")
    parser.add_argument("--batch_size_multimodal", type=int, default=20, help="Batch size for multimodal training (default: 20).")
    parser.add_argument("--num_epochs_multimodal", type=int, default=20, help="Number of epochs for multimodal training (default: 20).")
    parser.add_argument("--num_mc_samples", type=int, default=20, help="Number of Monte Carlo samples during training (default: 20).")
    parser.add_argument("--learning_rate_multimodal", type=float, default=0.001, help="Learning rate for multimodal training (default: 0.001).")
    parser.add_argument("--weight_decay_multimodal", type=float, default=1e-5, help="Weight decay for multimodal training (default: 1e-5).")
    parser.add_argument("--bathy_patch_base", type=int, default=30, help="Base size for bathymetry patches (default: 30).")
    parser.add_argument("--sss_patch_base", type=int, default=30, help="Base size for SSS patches (default: 30).")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes for classification (default: 10).") # Added num_classes
    parser.add_argument("--devices", type=str, default="cpu", help="Comma-separated list of devices (e.g., 'cuda:0,cuda:1' or 'cpu').") # Added devices

    args = parser.parse_args()

    # Define the fixed parameters within the CLI function
    const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",
        "moped_enable": True,
        "moped_delta": 0.1,
    }

    # You'll need to define `models_dir` if it's not a global or passed
    # For this example, let's assume `models_dir` is a subdirectory within `data_dir`
    models_dir = os.path.join(args.data_dir, "models")
    os.makedirs(models_dir, exist_ok=True) # Ensure models_dir exists

    model_paths = {
        "image": os.path.join(models_dir, "bayesian_model_type:image.pth"),
        "bathy": os.path.join(models_dir, "bayesian_model_type:bathy.pth"),
        "sss": os.path.join(models_dir, "bayesian_model_type:sss.pth"),
        "multimodal": os.path.join(models_dir, "_bayesian_model_type:multimodal.pth")
    }
    # Using the provided multimodal_model_path directly from the example


    optimizer_params = {
        "image_model": {"lr": 1e-5},
        "bathy_model": {"lr": 0.01},
        "sss_model": {"lr": 1e-5},
        "multimodal_model": {"lr": args.learning_rate_multimodal} # Use arg for multimodal LR
    }

    scheduler_params = {
        "image_model": {"step_size": 7, "gamma": 0.1},
        "bathy_model": {"step_size": 5, "gamma": 0.5},
        "sss_model": {"step_size": 7, "gamma": 0.7},
        "multimodal_model": {"step_size": 7, "gamma": 0.752}
    }

    training_params = {
        "num_epochs_unimodal": 30, # Not directly used by run_auv_training in this snippet but kept for completeness
        "num_epochs_multimodal": args.num_epochs_multimodal,
        "num_mc": args.num_mc_samples,
        "bathy_patch_base": f"patch_{args.bathy_patch_base}_bathy", # Format as string
        "sss_patch_base": f"patch_{args.sss_patch_base}_sss",     # Format as string
        "bathy_patch_types": ["patch_2_bathy", "patch_5_bathy", "patch_10_bathy", "patch_30_bathy", "patch_50_bathy"], # Example, could be args
        "sss_patch_types": ["patch_2_sss", "patch_5_sss", "patch_10_sss", "patch_30_sss", "patch_50_sss"],     # Example, could be args
        "batch_size_unimodal" : 8, # Example, could be arg
        "batch_size_multimodal" : args.batch_size_multimodal
    }

    # Convert devices string to a list of torch.device objects
    device_strings = [d.strip() for d in args.devices.split(',') if d.strip()]
    if not device_strings:
        devices = [torch.device("cpu")]
        print("No devices specified or invalid, defaulting to CPU.")
    else:
        devices = []
        for dev_str in device_strings:
            try:
                device = torch.device(dev_str)
                # Check if CUDA device is available and valid
                if 'cuda' in str(device) and not torch.cuda.is_available():
                    print(f"Warning: CUDA device {dev_str} not available. Skipping.")
                elif 'cuda' in str(device) and device.index is not None and device.index >= torch.cuda.device_count():
                    print(f"Warning: CUDA device {dev_str} index out of range. Skipping.")
                else:
                    devices.append(device)
            except RuntimeError as e:
                print(f"Warning: Could not create device from '{dev_str}': {e}. Skipping.")
        if not devices:
            print("No valid devices found after parsing. Defaulting to CPU.")
            devices = [torch.device("cpu")]
            
    print(f"Parsed devices: {devices}")

    # Pass the arguments to the run_auv_training function
    run_auv_training(
        optimizer_params=optimizer_params,
        scheduler_params=scheduler_params,
        training_params=training_params,
        root_dir=args.data_dir,
        devices=devices,
        const_bnn_prior_parameters=const_bnn_prior_parameters,
        num_classes=args.num_classes
    )

# --- CLI Function for Training From Scratch (from Example_training_from_scratch.py) ---
def training_from_scratch_cli():
    parser = argparse.ArgumentParser(
        description="Train a new Multimodal AUV model from scratch."
    )
    # Reconstructing arguments from your example command
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory for the training dataset.")
    parser.add_argument("--epochs_multimodal", type=int, default=20, help="Number of epochs for multimodal training (default: 20).")
    parser.add_argument("--num_mc", type=int, default=20, help="Number of Monte Carlo samples during training (default: 20).")
    parser.add_argument("--batch_size_multimodal", type=int, default=20, help="Batch size for multimodal training (default: 20).")
    parser.add_argument("--lr_multimodal", type=float, default=0.001, help="Learning rate for multimodal training (default: 0.001).")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes for the classification task (default: 10).")
    parser.add_argument("--devices", type=str, default="cpu", help="Comma-separated list of devices (e.g., 'cuda:0,cuda:1' or 'cpu').")
    parser.add_argument("--batch_size_unimodal", type=int, default=8, help="Batch size for unimodal training (default: 8).")
    parser.add_argument("--bathy_patch_base", type=int, default=30, help="Base size for bathymetry patches (default: 30).")
    parser.add_argument("--sss_patch_base", type=int, default=30, help="Base size for SSS patches (default: 30).")


    args = parser.parse_args()
    logging.info(f"Starting new model training from scratch with data from: {args.root_dir}")

    # Define parameters here to pass to the core function
    const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",
        "moped_enable": True,
        "moped_delta": 0.1,
    }

    optimizer_params = {
        "image_model": {"lr": 1e-5}, # Example fixed LR for unimodal
        "bathy_model": {"lr": 0.01}, # Example fixed LR for unimodal
        "sss_model": {"lr": 1e-5},   # Example fixed LR for unimodal
        "multimodal_model": {"lr": args.lr_multimodal}
    }

    scheduler_params = {
        "image_model": {"step_size": 7, "gamma": 0.1},
        "bathy_model": {"step_size": 5, "gamma": 0.5},
        "sss_model": {"step_size": 7, "gamma": 0.7},
        "multimodal_model": {"step_size": 7, "gamma": 0.752} # You might want to make this configurable too
    }

    training_params = {
        "num_epochs_unimodal": 30, # Example fixed value, consider making it an arg
        "num_epochs_multimodal": args.epochs_multimodal,
        "num_mc": args.num_mc,
        "bathy_patch_base": f"patch_{args.bathy_patch_base}_bathy", # Format as string
        "sss_patch_base": f"patch_{args.sss_patch_base}_sss",     # Format as string
        "bathy_patch_types": ["patch_2_bathy", "patch_5_bathy", "patch_10_bathy", "patch_30_bathy", "patch_50_bathy"],
        "sss_patch_types": ["patch_2_sss", "patch_5_sss", "patch_10_sss", "patch_30_sss", "patch_50_sss"],
        "batch_size_unimodal" : args.batch_size_unimodal,
        "batch_size_multimodal" : args.batch_size_multimodal
    }

    # Convert devices string to a list of torch.device objects
    device_strings = [d.strip() for d in args.devices.split(',') if d.strip()]
    if not device_strings:
        devices = [torch.device("cpu")]
        print("No devices specified or invalid, defaulting to CPU.")
    else:
        devices = []
        for dev_str in device_strings:
            try:
                device = torch.device(dev_str)
                # Check if CUDA device is available and valid
                if 'cuda' in str(device) and not torch.cuda.is_available():
                    print(f"Warning: CUDA device {dev_str} not available. Skipping.")
                elif 'cuda' in str(device) and device.index is not None and device.index >= torch.cuda.device_count():
                    print(f"Warning: CUDA device {dev_str} index out of range. Skipping.")
                else:
                    devices.append(device)
            except RuntimeError as e:
                print(f"Warning: Could not create device from '{dev_str}': {e}. Skipping.")
        if not devices:
            print("No valid devices found after parsing. Defaulting to CPU.")
            devices = [torch.device("cpu")]
            
    print(f"Parsed devices: {devices}")

    try:
        # Call your core function: run_AUV_training_from_scratch
        success = run_AUV_training_from_scratch(
            const_bnn_prior_parameters=const_bnn_prior_parameters,
            optimizer_params=optimizer_params,
            scheduler_params=scheduler_params,
            training_params=training_params,
            root_dir=args.root_dir,
            devices=devices,
            num_classes=args.num_classes
        )
        if not success:
            logging.error("New model training failed.")
            sys.exit(1)
        logging.info("New model training completed successfully!")
    except Exception as e:
        logging.exception(f"An unexpected error occurred during new model training: {e}")
        sys.exit(1)