import logging
import torch 
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from Multimodal_AUV.train.unimodal import train_unimodal_model, evaluate_unimodal_model 
from Multimodal_AUV.train.multimodal import train_multimodal_model, evaluate_multimodal_model
import os
from typing import Any , Optional, Dict, Tuple
from bayesian_torch.models.dnn_to_bnn import get_kl_loss

def define_optimizers_and_schedulers(
    models_dict: Dict[str, nn.Module],
    optimizer_params: Optional[Dict[str, Dict[str, Any]]] = None,
    scheduler_params: Optional[Dict[str, Dict[str, Any]]] = None,
    criterion_type: str = "cross_entropy"
) -> Tuple[nn.Module, Dict[str, optim.Optimizer], Dict[str, optim.lr_scheduler._LRScheduler]]:
    """
    Defines the loss, optimizers, and schedulers. Accepts optional hyperparameter overrides.

    Args:
        models_dict (dict): Dictionary of models.
        optimizer_params (dict, optional): Dict with keys as model names and values as optimizer kwargs.
        scheduler_params (dict, optional): Dict with keys as model names and values as scheduler kwargs.
        criterion_type (str): Type of loss criterion (default is "cross_entropy").

    Returns:
        Tuple of:
            - criterion
            - dict of optimizers
            - dict of schedulers
    """
    logging.info("Defining loss function and optimizers.")
    if criterion_type != "cross_entropy":
        logging.error(f"Unsupported criterion type provided: {criterion_type}")

    # 1. Loss
    if criterion_type == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported criterion: {criterion_type}")

    # 4. Build optimizers
    optimizers = {
        "image_model": optim.Adam(models_dict["image_model"].parameters(), **optimizer_params["image_model"]),
        "bathy_model": optim.Adam(models_dict["bathy_model"].parameters(), **optimizer_params["bathy_model"]),
        "sss_model": optim.Adam(models_dict["sss_model"].parameters(), **optimizer_params["sss_model"]),
        "multimodal_model": optim.Adam(
            models_dict["multimodal_model"].parameters(), 
            **optimizer_params["multimodal_model"]
        )
  
    }


    # 5. Build schedulers
    schedulers = {
        model_name: optim.lr_scheduler.StepLR(optimizers[model_name], **scheduler_params[model_name])
        for model_name in optimizers
    }

    return criterion, optimizers, schedulers

def train_and_evaluate_unimodal_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    num_epochs: int,
    device: torch.device,
    model_name: str,
    save_dir: str,
    num_mc: int,
    sum_writer: SummaryWriter
) -> None:
    """
    Train and evaluate a single unimodal PyTorch model for a specified number of epochs.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to be trained and evaluated.
    train_loader : DataLoader
        DataLoader providing the training dataset.
    test_loader : DataLoader
        DataLoader providing the test/validation dataset.
    criterion : nn.Module
        Loss function to optimize.
    optimizer : optim.Optimizer
        Optimizer used for training.
    scheduler : optim.lr_scheduler._LRScheduler
        Learning rate scheduler to update learning rate after each epoch.
    num_epochs : int
        Number of epochs to train the model.
    device : torch.device
        Device on which to run the training (CPU or GPU).
    model_name : str
        Identifier name for the model, used for logging and saving files.
    save_dir : str
        Directory path where model outputs and logs will be saved.
    num_mc : int
        Number of Monte Carlo samples for model uncertainty estimation.

    Returns
    -------
    None
        Trains the model, evaluates after each epoch, and logs progress.
    
    Logs key training parameters and progress, and calls training and evaluation helper functions.
    """
    logging.info("Starting training of unimodal model")
    logging.info(f"Model: {model.__class__.__name__}")
    logging.info(f"Number of epochs: {num_epochs}")
    logging.info(f"Device: {device}")
    logging.info(f"Model name: {model_name}")
    logging.info(f"Save directory: {save_dir}")
    logging.info(f"Number of Monte Carlo samples (num_mc): {num_mc}")
    logging.info(f"Training loader batch size: {getattr(train_loader, 'batch_size', 'unknown')}")
    logging.info(f"Test loader batch size: {getattr(test_loader, 'batch_size', 'unknown')}")
    logging.info(f"Criterion: {criterion.__class__.__name__}")
    logging.info(f"Optimizer: {optimizer.__class__.__name__}")
    logging.info(f"Scheduler: {scheduler.__class__.__name__}")    
    logging.info(f"Training started for unimodal model: {model_name}")
    for epoch in range(1, num_epochs):
    
        logging.debug(f"Epoch {epoch}/{num_epochs} - Training")
        train_accuracy, train_loss =train_unimodal_model(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch= epoch,
            total_num_epochs=num_epochs,
            num_mc=num_mc,
            device=device,
            model_type=model_name,
            csv_path=os.path.join(save_dir, f"{model_name}.csv"),
            sum_writer=sum_writer
        )
        
        logging.debug(f"Epoch {epoch}/{num_epochs} - Evaluation")
        val_accuracy = evaluate_unimodal_model(
            model=model,
            dataloader=test_loader,
            device=device,
            epoch=epoch,
            total_num_epochs=num_epochs,
            num_mc=num_mc,
            model_type=model_name,
            csv_path=os.path.join(save_dir, f"{model_name}_evaluate.csv"),
        )
        scheduler.step()
        sum_writer.add_scalar("train/loss/epoch", train_loss, epoch)
        sum_writer.add_scalar("val/accuracy/epoch", val_accuracy, epoch)

        logging.info(f"Training completed for unimodal model: {model_name}")
      

def train_and_evaluate_multimodal_model(
    train_loader: Any,
    test_loader: Any,
    multimodal_model: nn.Module,
    criterion: nn.CrossEntropyLoss,
    optimizer: optim.Optimizer,
    lr_scheduler: optim.lr_scheduler._LRScheduler,
    num_epochs: int,
    num_mc:int,
    device: torch.device,
    model_type: str,
    bathy_patch_type: str,
    sss_patch_type: str,
    csv_path: str, 
    sum_writer: SummaryWriter
) -> None:
    """
    Runs training and evaluation for a single multimodal model.

    Args:
        train_loader (Any): DataLoader for training.
        test_loader (Any): DataLoader for validation/testing.
        multimodal_model (nn.Module): The multimodal model to train and evaluate.
        criterion (nn.CrossEntropyLoss): Loss function.
        optimizer (optim.Optimizer): Optimizer for the multimodal model.
        lr_scheduler (optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        num_epochs (int): Number of epochs.
        device (torch.device): Device to use.
        model_type (str): Type of model (e.g., "multimodal").
        bathy_patch_type (str): Current bathy patch type.
        sss_patch_type (str): Current SSS patch type.
        csv_path (str): Path for saving results (e.g., logs, predictions).
    """
    logging.info("Starting multimodal model training and evaluation")
    logging.info(f"Model type: {model_type}")
    logging.info(f"bathy patch type: {bathy_patch_type}")
    logging.info(f"SSS patch type: {sss_patch_type}")
    logging.info(f"Number of epochs: {num_epochs}")
    logging.info(f"Number of Monte Carlo samples (num_mc): {num_mc}")
    logging.info(f"Device: {device}")
    logging.info(f"CSV path for logging results: {csv_path}")
    logging.info(f"Training loader batch size: {getattr(train_loader, 'batch_size', 'unknown')}")
    logging.info(f"Test loader batch size: {getattr(test_loader, 'batch_size', 'unknown')}")
    logging.info(f"Criterion: {criterion.__class__.__name__}")
    logging.info(f"Optimizer: {optimizer.__class__.__name__}")
    logging.info(f"Learning rate scheduler: {lr_scheduler.__class__.__name__}")
    logging.info(f"Model architecture: {multimodal_model.__class__.__name__}")

    for epoch in range(num_epochs): 
        logging.info(f"Epoch {epoch+1}/{num_epochs} - Multimodal model training started.")
        train_loss, train_accuracy =train_multimodal_model(
            multimodal_model=multimodal_model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch, 
            total_num_epochs=num_epochs, 
            device=device,
            model_type=model_type,
            num_mc=num_mc,
           bathy_patch_type=bathy_patch_type,
            sss_patch_type=sss_patch_type,
            csv_path=os.path.join(csv_path, f"multimodal_training.csv"),
            sum_writer=sum_writer
        )
        logging.info(f"Epoch {epoch+1}/{num_epochs} - Training complete.")

        lr_scheduler.step()
        val_accuracy = evaluate_multimodal_model(
            multimodal_model=multimodal_model,
            dataloader=test_loader,
            device=device,
            epoch=epoch, 
            total_num_epochs= num_epochs,
            num_mc =num_mc,
            csv_path=os.path.join(csv_path, f"multimodal_test.csv"),
            bathy_patch_type=bathy_patch_type,
            sss_patch_type=sss_patch_type,
            model_type=model_type
        )
        lr_scheduler.step()
        sum_writer.add_scalar("train/loss/epoch", train_loss, epoch)
        sum_writer.add_scalar("val/accuracy/epoch", val_accuracy, epoch)
        logging.info(f"Epoch {epoch+1}/{num_epochs} - Evaluation complete.")
    logging.info(f"Finished training & evaluation for multimodal model with C:{bathy_patch_type}, S:{sss_patch_type}")
