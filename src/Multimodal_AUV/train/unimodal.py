import logging
import torch 
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import csv
from typing import Optional
from bayesian_torch.models.dnn_to_bnn import get_kl_loss
from Multimodal_AUV.train.checkpointing import save_model
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay

import matplotlib
matplotlib.use('Agg') # This must be called *before* importing matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np
# Try to set a generic sans-serif font that is commonly available
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'Helvetica', 'Verdana']

def train_unimodal_model(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, total_num_epochs: int, num_mc: int,sum_writer: SummaryWriter,
          device: torch.device, model_type: str = "image", csv_path: str = "", patch_type: Optional[str] = None):
    """
    Trains a Bayesian neural network model for one epoch on the given dataset,
    calculates loss including KL divergence regularization, and logs training metrics.

    The function performs multiple stochastic forward passes per batch to
    approximate Bayesian inference, accumulates the average prediction and KL losses,
    then updates model weights accordingly.

    Training metrics including loss, accuracy, and learning rate are appended to a CSV file.

    Parameters
    ----------
    model : nn.Module
        The neural network model to be trained.
    dataloader : DataLoader
        DataLoader providing batches of data; each batch yields a tuple containing
        input tensors, labels, sonar patches, metadata, etc.
    criterion : nn.Module
        The loss function to compute prediction error (e.g., CrossEntropyLoss).
    optimizer : torch.optim.Optimizer
        The optimizer used to update model weights (e.g., Adam, SGD).
    epoch : int
        Current epoch number (zero-indexed).
    device : torch.device
        Device on which to perform training (CPU or GPU).
    model_type : str, optional
        Indicates input type for the model (default is "image").
    csv_path : str, optional
        File path to CSV where training metrics will be logged.
    patch_type : Optional[str], optional
        String indicating which patch tensor to use as model input (default is None).

    Returns
    -------
    None
        The function saves the model state to disk and logs training progress,
        but does not return any value.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    #Set the model to train 
    model.train()

    #Move the model to the device
    model.to(device)

    #Defining a KL weight to increase from 0 to 1 at final epoch
    kl_weight = (2 ** (epoch + 1)) / (2 ** total_num_epochs)

    #Check if the csv file exists 
    file_exists = os.path.isfile(csv_path)

    logging.info(f"Starting training epoch {epoch + 1}/{total_num_epochs} for model type: {model_type}")
    try:
        #Open the csv
        with open(csv_path, mode='a', newline='') as csvfile:
            #Define a writer for it
            writer = csv.writer(csvfile)
            #If it doesnt exist write in the headers 
            if not file_exists:
                writer.writerow(["Epoch", "Model type", "Loss", "Accuracy", "lr"])

            #Define parameters to keep track off
            total_loss, correct, total = 0, 0, 0
 
            #For each dataloader item extract all the data forms
            for i, batch in enumerate(dataloader):
                logging.info(f"Train batch {i+1}/{len(dataloader)} - Model: {model_type}")

                # Move core tensors to device
                inputs = batch["main_image"].to(device)
                labels = batch["label"].long().to(device)
                bathy_tensor = batch["bathy_image"].to(device)
                sss_image = batch["sss_image"].to(device)

                # Move all patch bathy and SSS patches to device dynamically
                patch_bathy = {k: v.to(device) for k, v in batch.get("patch_bathy", {}).items()}
                patch_sss = {k: v.to(device) for k, v in batch.get("patch_sss", {}).items()}

                # Merge both patch sources into a single dictionary
                all_patches = {**patch_bathy, **patch_sss, "patch_30_bathy": bathy_tensor, "patch_30_sss": sss_image}

                # Select patch tensor if type is provided
                patch = all_patches.get(patch_type) if patch_type else None

                #Zero the optimiser
                optimizer.zero_grad()

                #The move the inputs to the contiguous memory
                if model_type == "image":
                    model_input = inputs
                elif model_type == "sss":
                    model_input = sss_image
                elif model_type == "bathy":
                    model_input = bathy_tensor
                else:
                    # fallback or raise error if unexpected model_type
                    logging.error(f"Unknown model_type: {model_type}") 
                    raise ValueError(f"Unknown model_type: {model_type}")

                #Does monte carlo runs passing the inputs through the model, appending these to outputs and getting the kl loss for each pass
                outputs, kl_losses = [], []

                for _ in range(num_mc):
                    out = model(model_input)
                    outputs.append(out)
                    kl_losses.append(get_kl_loss(model))

                #Get the average output
                output = torch.mean(torch.stack(outputs), dim=0)

                #Get the average, scaled KL loss
                scaled_kl = torch.mean(torch.stack(kl_losses), dim=0) / dataloader.batch_size

                #Get the CE loss from the outputs
                cross_entropy_loss = criterion(output, labels)

                #Combine the losses
                loss = cross_entropy_loss + (kl_weight * scaled_kl)

                #Backwards pass and optimiser step
                loss.backward()
                optimizer.step()

                #Get the predicted label
                output = output.float()
                _, predicted = output.max(1)
                #Append the loss to total loss
                total_loss += loss.item()
                #Get the correct predictions and total number of units
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                sum_writer.add_scalar("Loss/train", loss, i)
                
            #Estimate the train accuracy, loss and learnign rate
            train_accuracy = correct / total
            train_loss = total_loss / total
            lr = optimizer.param_groups[0]['lr']

            #Log this
            logging.info(f"Epoch {epoch + 1} | Final Loss: {train_loss:.4f} | Accuracy: {train_accuracy:.4f} | LR: {lr:.6f}")
            writer.writerow([epoch + 1, model_type, train_loss, train_accuracy, lr])

        #Save the model every 5 epochs:
        if epoch % 5 ==0:
            save_model(model, csv_path, model_type)

    except:
        save_model(model, csv_path, model_type)
        logging.error(f"Error at epoch {epoch}", exc_info=True)
        train_accuracy, train_loss= 0.0, 0.0
    return train_accuracy, train_loss


def evaluate_unimodal_model(model: nn.Module, dataloader: DataLoader, device: torch.device, epoch: int, csv_path: str, total_num_epochs: int, num_mc: int, 
             model_type: str = "image", patch_type: Optional[str] = None):
    """
    Evaluate a single model using Monte Carlo dropout to estimate predictive uncertainty,
    calculate loss and accuracy on the provided dataset, and log results to a CSV file.

    Args:
        model (nn.Module): The PyTorch model to evaluate.
        dataloader (DataLoader): DataLoader providing batches of test data.
        device (torch.device): Device to perform computations on (CPU or GPU).
        epoch (int): Current epoch number, used for logging and KL weight calculation.
        csv_path (str): File path to save evaluation metrics in CSV format.
        model_type (str, optional): Type of model input to use (default is "image").
        patch_type (str or None, optional): Type of patch input for the model, if any.

    Returns:
        None: Results are printed and appended to the CSV file.
    """
    #Set model to train to ensure Monte Carlo dropout consistency
    model.train()  

    #Define the criterion
    criterion = nn.CrossEntropyLoss()

    #Define the KL weight, this increase from 0 to 1 at final epoch
    kl_weight = (2 ** (epoch + 1)) / (2 ** total_num_epochs)

    #Check file exists
    file_exists = os.path.isfile(csv_path)

    logging.info(f"Starting evaluation for epoch {epoch + 1} (model type: {model_type})")
    try:
        #Open csv
        with open(csv_path, mode='a', newline='') as csvfile:
            #Set this csv as the writer
            writer = csv.writer(csvfile)
                #If it didnt exist then write the headers
            if not file_exists:
                    writer.writerow(["Epoch", "Model Type", "Test Loss", "Test Accuracy", "predictive_uncertainty", "model_uncertainty"])
       

                #Define some metrics to keep track off
            correct, total, total_loss = 0, 0, 0
            all_predictive_uncertainties = []
            all_aleatoric_uncertainties = []
            all_predicted = []
            all_labels = []
                #For each dataloader item extract all the data forms
            for i, batch in enumerate(dataloader):
                logging.info(f"Train batch {i+1}/{len(dataloader)} - Model: {model_type}")
                # Move core tensors to device
                inputs = batch["main_image"].to(device)
                labels = batch["label"].long().to(device)
                bathy_tensor = batch["bathy_image"].to(device)
                sss_image = batch["sss_image"].to(device)

                # Move all patch bathy and SSS patches to device dynamically
                patch_bathy = {k: v.to(device) for k, v in batch.get("patch_bathy", {}).items()}
                patch_sss = {k: v.to(device) for k, v in batch.get("patch_sss", {}).items()}

                # Merge both patch sources into a single dictionary
                all_patches = {**patch_bathy, **patch_sss, "patch_30_bathy": bathy_tensor, "patch_30_sss": sss_image}

                # Select patch tensor if type is provided
                patch = all_patches.get(patch_type) if patch_type else None
                #Move required data to contiguous memory
                if model_type == "image":
                    model_input = inputs
                elif model_type == "sss":
                    model_input = sss_image
                elif model_type == "bathy":
                    model_input = bathy_tensor
                else:
                    # fallback or raise error if unexpected model_type
                    logging.error(f"Unknown model_type: {model_type}")
                    raise ValueError(f"Unknown model_type: {model_type}")
            
                #Define some metrics to keep track of MC runs
                outputs_mc_logits, kl_losses = [], []

                #Does  monte carlo runs passing the inputs through the model, appending these to outputs and getting the kl loss for each pass
                for _ in range(num_mc):
                    out_logits = model(model_input) # Get logits
                    outputs_mc_logits.append(out_logits)
                    kl_losses.append(get_kl_loss(model)) # KL loss for that sample

           

                # 1. Calculate Average Output (for prediction and cross-entropy loss)
                # This is the mean of the logits from MC samples
                output_mean_logits = torch.mean(torch.stack(outputs_mc_logits), dim=0)
                # Convert mean logits to probabilities for cross-entropy, if criterion expects it

                # 2. Calculate Average, Scaled KL Loss (for logging/total loss if applicable during eval)
                scaled_kl = torch.mean(torch.stack(kl_losses), dim=0) / dataloader.batch_size # Assuming batch_size scale

                # 3. Calculate Cross-Entropy Loss (using the average output)
                cross_entropy_loss = criterion(output_mean_logits, labels)

                # 4. Combine losses (KL term usually only for training, but included for completeness)
                loss = cross_entropy_loss + (kl_weight * scaled_kl)

                # 5. Get the predicted label from the averaged output
                # Convert logits to probabilities, then get max for prediction
                probabilities_mean = torch.softmax(output_mean_logits, dim=-1)
                _, predicted = probabilities_mean.max(1)

                # Accumulate loss, correct prdictions, and total count
                total_loss += loss.item()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                # --- Uncertainty Calculations ---
                # Create a stacked tensor of probabilities from all MC samples
                # This is crucial: prob_outputs_mc needs to be defined from outputs_mc_logits
                prob_outputs_list = [torch.softmax(out_logits, dim=-1) for out_logits in outputs_mc_logits]
                prob_outputs_mc = torch.stack(prob_outputs_list, dim=0)
            
                # Calculate Epistemic Uncertainty (Variance of probabilities across MC samples)

                variance_per_class = torch.var(prob_outputs_mc, dim=0) # Shape: (batch_size, num_classes)
                # Average variance across classes for each batch item
                epistemic_uncertainty_batch = variance_per_class.mean(dim=1) # Shape: (batch_size,)
                all_predictive_uncertainties.extend(epistemic_uncertainty_batch.cpu().detach().numpy())

                # Calculate Aleatoric Uncertainty (Average Entropy of individual MC sample probabilities)
                # Entropy for each (MC_sample, batch_item)
                epsilon = 1e-7
                entropy_per_mc_and_item = -torch.sum(prob_outputs_mc * torch.log(prob_outputs_mc + epsilon), dim=-1) # Shape: (num_mc, batch_size)
                # Average across MC samples for each batch item
                aleatoric_uncertainty_batch = torch.mean(entropy_per_mc_and_item, dim=0) # Shape: (batch_size,)
                all_aleatoric_uncertainties.extend(aleatoric_uncertainty_batch.cpu().detach().numpy())
                all_predicted.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            # Calculate epoch-level averages for loss, accuracy, and uncertainties
            accuracy = correct / total
            avg_loss = total_loss / total

            # Average the collected uncertainties over the entire dataset
            avg_predictive_uncertainty = np.mean(all_predictive_uncertainties) if all_predictive_uncertainties else 0.0
            avg_aleatoric_uncertainty = np.mean(all_aleatoric_uncertainties) if all_aleatoric_uncertainties else 0.0
           

            # Print and save to CSV
            logging.info(
                f"Eval Epoch {epoch + 1} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f} | "
                f"Epistemic UQ: {avg_predictive_uncertainty:.6f} | Aleatoric UQ: {avg_aleatoric_uncertainty:.6f}"
            )
            try: # Inner try for plotting
                        # Confusion Matrix
                        cm = confusion_matrix(all_labels, all_predicted) # Consider adding labels=list(range(num_classes))
                        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                        fig, ax = plt.subplots(figsize=(8, 8))
                        disp.plot(cmap="Blues", ax=ax)
                        plt.title(f"Confusion Matrix for Epoch {epoch}")
                        # Get the parent directory of the CSV path
                        parent_path = os.path.dirname(csv_path)
                        # Define the subfolder path for confusion matrices
                        conf_matrix_folder = os.path.join(parent_path, "confusion_matrices")
                        # Create the folder if it doesn't exist
                        os.makedirs(conf_matrix_folder, exist_ok=True)
                        # Create the filename and full path
                        matrix_filename = f"conf_matrix_model_{model_type}_{epoch}.png"
                        matrix_path = os.path.join(conf_matrix_folder, matrix_filename)
                        # Save the plot
                        plt.savefig(matrix_path)
                        # Log the full path where it's saved
                        logging.info(f"Confusion matrix saved to: {matrix_path}")
            except Exception as e:
                        logging.warning(f"Confusion matrix not saved due to plotting error: {e}", exc_info=True)
            finally: # Ensures fig is closed regardless of success or failure
                        if fig is not None:
                            plt.close(fig)
            #Write this
            writer.writerow([
                epoch + 1,
                model_type,
                avg_loss,
                accuracy,
                avg_predictive_uncertainty,
                avg_aleatoric_uncertainty 
            ])
    except:
            save_model(model, csv_path, model_type)
            logging.error(f"Error at epoch {epoch}", exc_info=True)
            accuracy =0.0
    return accuracy
