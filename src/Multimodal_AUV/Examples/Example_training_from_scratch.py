import logging
from typing import Dict, Any
import datetime
import os
import sys
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse

# Defined in project
from Multimodal_AUV.models.model_utils import define_models
from Multimodal_AUV.utils.device import move_models_to_device, check_model_devices
from Multimodal_AUV.config.paths import setup_environment_and_devices
from Multimodal_AUV.data.loaders import prepare_datasets_and_loaders
from Multimodal_AUV.train.loop_utils import train_and_evaluate_unimodal_model, train_and_evaluate_multimodal_model, define_optimizers_and_schedulers
from Multimodal_AUV.train.checkpointing import load_and_fix_state_dict
from Multimodal_AUV.inference.inference_data import prepare_inference_datasets_and_loaders
from Multimodal_AUV.inference.predictors import multimodal_predict_and_save
from Multimodal_AUV.config.paths import get_empty_gpus

def main(
    const_bnn_prior_parameters: Dict[str, Any],
    optimizer_params: Dict[str, Dict[str, Any]],
    scheduler_params: Dict[str, Dict[str, Any]],
    training_params: Dict[str, Any],
    root_dir: str,
    devices: torch.device 
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
    _, _, multimodal_train_loader, multimodal_test_loader, num_classes, _ = prepare_datasets_and_loaders(root_dir, batch_size_unimodal=training_params["batch_size_unimodal"], batch_size_multimodal=training_params["batch_size_multimodal"])
    logging.info(f"Number of classes: {num_classes}")

    # 3. Model Definition and Initialization
    logging.info("Defining models...")
    models_dict = define_models(device=devices[0], num_classes=num_classes, const_bnn_prior_parameters=const_bnn_prior_parameters)
   
    models_dict = move_models_to_device(models_dict, devices, use_multigpu_for_multimodal=True)
    logging.info("Models defined and will be moved to device during training/inference as needed.")
    torch.cuda.empty_cache()

    # 4. Optimizers and Schedulers
    logging.info("Setting up criterion, optimizers and schedulers...")
    criterion, optimizers, schedulers = define_optimizers_and_schedulers(models_dict, optimizer_params, scheduler_params)
    
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
        device=devices[0],
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

    #Set up args 
    parser = argparse.ArgumentParser(description="Train a multimodal AUV model.")
    parser.add_argument('--root_dir', type=str, default='./data/',
                        help='Root directory for datasets and outputs.')
    parser.add_argument('--epochs_multimodal', type=int, default=30,
                        help='Number of epochs for multimodal model training.')
    parser.add_argument('--num_mc', type=int, default=5,
                        help='Number of Monte Carlo samples for Bayesian models.')
    parser.add_argument('--batch_size_multimodal', type=int, default=12,
                        help='Batch size for multimodal training.')
    parser.add_argument('--lr_multimodal', type=float, default=5e-5,
                        help='Learning rate for multimodal model optimizer.')


    args = parser.parse_args()

   
     # Setup devices based on your utility function (using get_empty_gpus as defined or imported)
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

    #Defining bayesian model initialisation parameters
    const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",
        "moped_enable": True,
        "moped_delta": 0.1,
    }

    #Defining optimiser and scheduler parameres
    optimizer_params = {
        "image_model": {"lr": 1e-5},
        "bathy_model": {"lr": 0.01},
        "sss_model": {"lr": 1e-5},
        "multimodal_model": {"lr": args.lr_multimodal}
    }

    scheduler_params = {
        "image_model": {"step_size": 7, "gamma": 0.1},
        "bathy_model": {"step_size": 5, "gamma": 0.5},
        "sss_model": {"step_size": 7, "gamma": 0.7},
        "multimodal_model": {"step_size": 7, "gamma": 0.752}
    }

    #DEfining train parameters (note a lot of redundent in this)
    training_params = {
        "num_epochs_unimodal": 1,
        "num_epochs_multimodal": args.epochs_multimodal,
        "num_mc": args.num_mc,
        "bathy_patch_base": "patch_30_bathy",
        "sss_patch_base": "patch_30_sss",
        "bathy_patch_types": ["patch_2_bathy", "patch_5_bathy", "patch_10_bathy", "patch_30_bathy", "patch_50_bathy"],
        "sss_patch_types": ["patch_2_sss", "patch_5_sss", "patch_10_sss", "patch_30_sss", "patch_50_sss"],
        "batch_size_unimodal": 1,
        "batch_size_multimodal": args.batch_size_multimodal
    }

    #Call the main
    main(
        const_bnn_prior_parameters,
        optimizer_params,
        scheduler_params,
        training_params,
        args.root_dir,
        devices
    )