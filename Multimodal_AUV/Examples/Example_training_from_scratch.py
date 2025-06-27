import logging
from typing import Dict, Any
import datetime
import os
import sys
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse

# Assuming these imports are correctly set up in your project
from Multimodal_AUV.models.model_utils import define_models
from Multimodal_AUV.utils.device import move_models_to_device, check_model_devices
from Multimodal_AUV.config.paths import setup_environment_and_devices
from Multimodal_AUV.data.loaders import prepare_datasets_and_loaders
from Multimodal_AUV.train.loop_utils import train_and_evaluate_unimodal_model, train_and_evaluate_multimodal_model, define_optimizers_and_schedulers
from Multimodal_AUV.train.checkpointing import load_and_fix_state_dict
from Multimodal_AUV.inference.inference_data import prepare_inference_datasets_and_loaders
from Multimodal_AUV.inference.predictors import multimodal_predict_and_save

def main(
    const_bnn_prior_parameters: Dict[str, Any],
    model_paths: Dict[str, str],
    multimodal_model_path: str,
    optimizer_params: Dict[str, Dict[str, Any]],
    scheduler_params: Dict[str, Dict[str, Any]],
    training_params: Dict[str, Any],
    root_dir: str,
    models_dir: str,
    strangford_dir: str,
    mulroy_dir: str,
    devices: torch.device # Changed to a single device for consistency with the original code, can be adjusted for multi-GPU
):
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Setup file logging
    log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "training.log")

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Setup console logging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Initialize TensorBoard writer
    tb_log_dir = os.path.join("tensorboard_logs", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    sum_writer = SummaryWriter(log_dir=tb_log_dir)
    sum_writer.add_text("Init", "TensorBoard logging started", 0)

    logging.info("Logging initialized.")

    # 1. Environment and Device Setup
    logging.info("Setting up environment and devices...")
    logging.info(f"Using device: {str(devices)}") # Adjusted for single device
    # 2. Dataset and DataLoader Preparation
    logging.info("Preparing datasets and data loaders...")
    unimodal_train_loader, unimodal_test_loader, multimodal_train_loader, multimodal_test_loader, num_classes, dataset = prepare_datasets_and_loaders(root_dir, batch_size_unimodal=training_params["batch_size_unimodal"], batch_size_multimodal=training_params["batch_size_multimodal"])
    logging.info(f"Number of classes: {num_classes} | Dataset split: {len(unimodal_train_loader.dataset)} training samples, {len(unimodal_test_loader.dataset)} test samples")

    # 3. Model Definition and Initialization
    logging.info("Defining models...")
    models_dict = define_models(model_paths, device=devices, num_classes=num_classes, const_bnn_prior_parameters=const_bnn_prior_parameters)
    # The original code had a commented out line for move_models_to_device.
    # If you intend to use multiple GPUs, you'll need to uncomment and configure it.
    # models_dict = move_models_to_device(models_dict, devices, use_multigpu_for_multimodal=True)
    logging.info("Models defined and will be moved to device during training/inference as needed.")
    torch.cuda.empty_cache()

    # 4. Optimizers and Schedulers
    logging.info("Setting up criterion, optimizers and schedulers...")
    criterion, optimizers, schedulers = define_optimizers_and_schedulers(models_dict, optimizer_params, scheduler_params)

    # 5. Train Unimodal Models (uncomment if needed)
    # model_labels = {
    #     "image_model": "image",
    #     "bathy_model": "bathy",
    #     "sss_model": "sss",
    # }
    # logging.info("Starting training of unimodal models...")
    # print("Starting training of unimodal models...")
    # for model_key in ["image_model", "bathy_model", "sss_model"]:
    #     print(f"training model: {model_key}")
    #     logging.info(f"Training {model_key}...")
    #     train_and_evaluate_unimodal_model(
    #         model=models_dict[model_key],
    #         train_loader=unimodal_train_loader,
    #         test_loader=unimodal_test_loader,
    #         criterion=criterion,
    #         optimizer=optimizers[model_key],
    #         scheduler=schedulers[model_key],
    #         num_epochs=training_params["num_epochs_unimodal"],
    #         num_mc=training_params["num_mc"],
    #         device=devices, # Changed to single device
    #         model_name=model_labels[model_key],
    #         save_dir=f"{root_dir}csvs/",
    #         sum_writer=sum_writer
    #     )
    #     logging.info(f"Finished training {model_key}.")

    # 6. Load and Check Multimodal Model
    logging.info("Attempting to load multimodal model...")

    # Load and fix state dictionary, if a path is provided and it exists.
    # For training from scratch, you might want to skip this or handle it differently.
    if multimodal_model_path and os.path.exists(multimodal_model_path):
        if load_and_fix_state_dict(models_dict['multimodal_model'], multimodal_model_path, devices):
            logging.info("Multimodal model loaded successfully.")
        else:
            logging.warning("Multimodal model loading failed or file not found at specified path. Training from scratch.")
    else:
        logging.info("No pre-trained multimodal model path provided or path does not exist. Training multimodal model from scratch.")


    if not check_model_devices(models_dict['multimodal_model'], devices):
        logging.error("Multimodal model is not on expected device.")
        # Decide if you want to exit or try to move it. For now, we'll exit.
        sys.exit(1)


    # 7. Run Base Multimodal Training
    logging.info("Starting base multimodal training...")
    print("Starting base multimodal training...")
    train_and_evaluate_multimodal_model(
        train_loader=multimodal_train_loader,
        test_loader=multimodal_test_loader,
        multimodal_model=models_dict["multimodal_model"],
        criterion=criterion,
        optimizer=optimizers["multimodal_model"],
        lr_scheduler=schedulers["multimodal_model"],
        num_epochs=training_params["num_epochs_multimodal"],
        device=devices, # Changed to single device
        model_type="multimodal",
        bathy_patch_type=training_params["bathy_patch_base"],
        sss_patch_type=training_params["sss_patch_base"],
        csv_path=f"{root_dir}csvs/",
        num_mc=training_params["num_mc"],
        sum_writer=sum_writer
    )
    logging.info("Base multimodal training complete.")

    sum_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a multimodal AUV model.")
    parser.add_argument('--root_dir', type=str, default='./data/',
                        help='Root directory for datasets and outputs.')
    parser.add_argument('--models_dir', type=str, default='./models/',
                        help='Directory to save and load models.')
    parser.add_argument('--strangford_dir', type=str, default='./data/Strangford/',
                        help='Directory for Strangford dataset.')
    parser.add_argument('--mulroy_dir', type=str, default='./data/Mulroy/',
                        help='Directory for Mulroy dataset.')
    parser.add_argument('--multimodal_model_path', type=str, default=None,
                        help='Path to a pre-trained multimodal model. If not provided or file does not exist, training will start from scratch.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (e.g., "cuda:0" or "cpu").')
    parser.add_argument('--epochs_unimodal', type=int, default=30,
                        help='Number of epochs for unimodal model training.')
    parser.add_argument('--epochs_multimodal', type=int, default=30,
                        help='Number of epochs for multimodal model training.')
    parser.add_argument('--num_mc', type=int, default=12,
                        help='Number of Monte Carlo samples for Bayesian models.')
    parser.add_argument('--batch_size_unimodal', type=int, default=8,
                        help='Batch size for unimodal training.')
    parser.add_argument('--batch_size_multimodal', type=int, default=12,
                        help='Batch size for multimodal training.')
    parser.add_argument('--lr_image', type=float, default=1e-5,
                        help='Learning rate for image model optimizer.')
    parser.add_argument('--lr_bathy', type=float, default=0.01,
                        help='Learning rate for bathymetry model optimizer.')
    parser.add_argument('--lr_sss', type=float, default=1e-5,
                        help='Learning rate for SSS model optimizer.')
    parser.add_argument('--lr_multimodal', type=float, default=5e-5,
                        help='Learning rate for multimodal model optimizer.')


    args = parser.parse_args()

    # Setup environment and devices using the parsed arguments
    # Note: setup_environment_and_devices might still return a device based on its internal logic
    # but we'll prioritize the argparse `device` for model placement.
    # Ensure setup_environment_and_devices is compatible with accepting these paths or adjust its role.
    # For simplicity, assuming setup_environment_and_devices just returns the device
    # and other paths are handled by passing them directly to main.
    # If setup_environment_and_devices actually *sets* these paths, you'll need to adapt it.
    device = torch.device(args.device)

    # Re-define model_paths to use the models_dir from argparse
    model_paths = {
        "image": os.path.join(args.models_dir, "bayesian_model_type:image.pth"),
        "bathy": os.path.join(args.models_dir, "bayesian_model_type:bathy.pth"),
        "sss": os.path.join(args.models_dir, "bayesian_model_type:sss.pth"),
        "multimodal": os.path.join(args.models_dir, "_bayesian_model_type:multimodal.pth")
    }

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
        "image_model": {"lr": args.lr_image},
        "bathy_model": {"lr": args.lr_bathy},
        "sss_model": {"lr": args.lr_sss},
        "multimodal_model": {"lr": args.lr_multimodal}
    }

    scheduler_params = {
        "image_model": {"step_size": 7, "gamma": 0.1},
        "bathy_model": {"step_size": 5, "gamma": 0.5},
        "sss_model": {"step_size": 7, "gamma": 0.7},
        "multimodal_model": {"step_size": 7, "gamma": 0.752}
    }

    training_params = {
        "num_epochs_unimodal": args.epochs_unimodal,
        "num_epochs_multimodal": args.epochs_multimodal,
        "num_mc": args.num_mc,
        "bathy_patch_base": "patch_30_bathy",
        "sss_patch_base": "patch_30_sss",
        "bathy_patch_types": ["patch_2_bathy", "patch_5_bathy", "patch_10_bathy", "patch_30_bathy", "patch_50_bathy"],
        "sss_patch_types": ["patch_2_sss", "patch_5_sss", "patch_10_sss", "patch_30_sss", "patch_50_sss"],
        "batch_size_unimodal": args.batch_size_unimodal,
        "batch_size_multimodal": args.batch_size_multimodal
    }

    main(
        const_bnn_prior_parameters,
        model_paths,
        args.multimodal_model_path, # Use the path from argparse
        optimizer_params,
        scheduler_params,
        training_params,
        args.root_dir,
        args.models_dir,
        args.strangford_dir,
        args.mulroy_dir,
        device
    )