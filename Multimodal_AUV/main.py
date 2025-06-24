import logging
from typing import Dict, Any
import datetime
import os
import sys
import torch
from torch.utils.tensorboard import SummaryWriter
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
    training_params: Dict[str, Any],  root_dir, models_dir, strangford_dir, mulroy_dir, devices
):
    

    # Setup logging once
    log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "training.log")

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s | %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
        
    # Initialize TensorBoard writer
    tb_log_dir = os.path.join("tensorboard_logs", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    sum_writer = SummaryWriter(log_dir=tb_log_dir)
    sum_writer.add_text("Init", "TensorBoard logging started", 0)
    #In terminal type:
    #tensorboard --logdir=tensorboard_logs --port=6006
    #In broswer navigate to:
    #http://localhost:6006

    # Example use
    logging.info("Logging initialized.")

    # 1. Environment and Device Setup
    logging.info("Setting up environment and devices...")
    logging.info(f"Using devices: {[str(d) for d in devices]}")
    # 2. Dataset and DataLoader Preparation
    logging.info("Preparing datasets and data loaders...")
    unimodal_train_loader, unimodal_test_loader, multimodal_train_loader, multimodal_test_loader, num_classes, dataset = prepare_datasets_and_loaders(root_dir, batch_size_unimodal=training_params["batch_size_unimodal"], batch_size_multimodal=training_params["batch_size_multimodal"])
    logging.info(f"Number of classes: {num_classes} | Dataset split: {len(unimodal_train_loader.dataset)} training samples, {len(unimodal_test_loader.dataset)} test samples")

    # 3. Model Definition and Initialization
    logging.info("Defining models...")
    models_dict = define_models(model_paths, device=device, num_classes=num_classes, const_bnn_prior_parameters=const_bnn_prior_parameters)
    models_dict = move_models_to_device(models_dict, devices, use_multigpu_for_multimodal=True)
    logging.info("Models moved to devices.")
    torch.cuda.empty_cache()

    # 4. Optimizers and Schedulers
    logging.info("Setting up criterion, optimizers and schedulers...")
    criterion, optimizers, schedulers = define_optimizers_and_schedulers(models_dict, optimizer_params, scheduler_params)

    # 5. Train Unimodal Models
    model_labels = {
        "image_model": "image",
        "bathy_model": "bathy",
        "sss_model": "sss",
    }
    logging.info("Starting training of unimodal models...")
    print("Starting training of unimodal models...")
    for model_key in ["image_model", "bathy_model", "sss_model"]:
       print(f"training model: {model_key}")
       logging.info(f"Training {model_key}...")
       train_and_evaluate_unimodal_model(
           model=models_dict[model_key],
           train_loader=unimodal_train_loader,
           test_loader=unimodal_test_loader,
           criterion=criterion,
           optimizer=optimizers[model_key],
           scheduler=schedulers[model_key],
           num_epochs=training_params["num_epochs_unimodal"],
           num_mc=training_params["num_mc"],
           device=devices[0],
           model_name=model_labels[model_key], 
           save_dir=f"{root_dir}csvs/",
           sum_writer=sum_writer
       )
       logging.info(f"Finished training {model_key}.")


   # 6. Load and Check Multimodal Model
    logging.info("Attempting to load multimodal model...")

    if load_and_fix_state_dict(models_dict['multimodal_model'], multimodal_model_path, devices[0]):
      logging.info("Multimodal model loaded successfully.")
    else:
      logging.warning("Multimodal model loading failed or file not found.")

    if not check_model_devices(models_dict['multimodal_model'], devices[0]):
       logging.error("Multimodal model is not on expected device.")
       return

    #7. Run Base Multimodal Training
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
    csv_path=f"{root_dir}csvs/multimodal_results.csv",
    num_mc=training_params["num_mc"],
    sum_writer=sum_writer
    )
    logging.info("Base multimodal training complete.")


    #Check and its running up to here
 
    #8. Run All Combinations of Multimodal Patch Types
    #logging.info("Starting grid search over patch types...")

    #for bathy_patch_type in training_params["bathy_patch_types"]:
    #    for sss_patch_type in training_params["sss_patch_types"]:
    #        logging.info(f"Training: Bathy Patch = {bathy_patch_type}, SSS Patch = {sss_patch_type}")
    #        print(f"\n--- Starting multimodal Training for: Bathy: {bathy_patch_type}, SSS: {sss_patch_type} ---")
    #        train_and_evaluate_multimodal_model(
    #            train_loader = multimodal_train_loader,
    #            test_loader = multimodal_test_loader,
    #            multimodal_model = models_dict["multimodal_model"],
    #            criterion = criterion,
    #            optimizer = optimizers["multimodal_model"],
    #            lr_scheduler = schedulers["multimodal_model"], 
    #            num_epochs = training_params["num_epochs_multimodal"],
    #            device = devices[1], 
    #            model_type = "multimodal", 
    #            bathy_patch_type = bathy_patch_type,
    #            sss_patch_type= sss_patch_type,
    #            csv_path=f"{root_dir}multimodal_results_{bathy_patch_type}_{sss_patch_type}.csv",
    #            num_mc = training_params["num_mc"],
    #            sum_writer=sum_writer
    #        )
    #        logging.info(f"Finished training: bathy = {bathy_patch_type}, SSS = {sss_patch_type}")
    #        print(f"--- Finished multimodal Training for: bathy: {bathy_patch_type}, SSS: {sss_patch_type} ---")
    #        torch.cuda.empty_cache()

    # 9. Final Inference
    logging.info("Starting final inference across survey datasets...")
    dataloader_whole_survey = prepare_inference_datasets_and_loaders(strangford_dir, mulroy_dir, training_params["batch_size_unimodal"])
    multimodal_predict_and_save(
        multimodal_model = models_dict['multimodal_model'],
        dataloader = dataloader_whole_survey,
        device = devices[0],
        csv_path=f"{root_dir}whole_survey_resulrs.csv",
        num_mc_samples= training_params["num_mc"],
        sss_patch_type="",
        channel_patch_type="",
        model_type="multimodal"
    )
    logging.info("Final inference complete. Results saved.")
if __name__ == "__main__":
    root_dir, models_dir, strangford_dir, mulroy_dir, device = setup_environment_and_devices()

    const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",
        "moped_enable": True,
        "moped_delta": 0.1,
    }
    model_paths = {
    "image": os.path.join(models_dir, "bayesian_model_type:image.pth"),
    "bathy": os.path.join(models_dir, "bayesian_model_type:bathy.pth"),
    "sss": os.path.join(models_dir, "bayesian_model_type:sss.pth"),
    "multimodal": os.path.join(models_dir, "_bayesian_model_type:multimodal.pth")
    }
    multimodal_model_path = os.path.join(models_dir, "teacher_model_epoch_19.pth")

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

    training_params = {
        "num_epochs_unimodal": 30,
        "num_epochs_multimodal": 30,
        "num_mc":7,
        "bathy_patch_base": "patch_30_bathy",
        "sss_patch_base": "patch_30_sss",
        "bathy_patch_types": ["patch_2_bathy", "patch_5_bathy", "patch_10_bathy", "patch_30_bathy", "patch_50_bathy"],
        "sss_patch_types": ["patch_2_sss", "patch_5_sss", "patch_10_sss", "patch_30_sss", "patch_50_sss"],
        "batch_size_unimodal" : 8,
        "batch_size_multimodal" : 4
    }

    main(
        const_bnn_prior_parameters,
        model_paths,
        multimodal_model_path,
        optimizer_params,
        scheduler_params,
        training_params,  root_dir, models_dir, strangford_dir, mulroy_dir, device
    )