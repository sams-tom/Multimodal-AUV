import os
import csv
import tempfile
import unittest
from unittest import mock
from unittest.mock import patch, mock_open, MagicMock
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib
matplotlib.use('Agg') # MUST be called BEFORE importing pyplot
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset

from bayesian_torch.models.dnn_to_bnn import get_kl_loss

from Multimodal_AUV.train.unimodal import evaluate_unimodal_model, train_unimodal_model
from Multimodal_AUV.train.multimodal import evaluate_multimodal_model, train_multimodal_model
from Multimodal_AUV.train.loop_utils import define_optimizers_and_schedulers, train_and_evaluate_unimodal_model, train_and_evaluate_multimodal_model
from Multimodal_AUV.train.checkpointing import save_model, load_and_fix_state_dict

# Mocks and dummy classes
def mock_kl_loss(model):
    return torch.tensor(0.05)

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(150528, 2)  # Match flattened image size

    def forward(self, x):
        assert x.numel() == 150528, f"Expected input with 150528 elements but got {x.numel()}"
        return self.linear(x.view(x.size(0), -1))

class DummyBayesianModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Define layers for each input type
        self.main_linear = torch.nn.Linear(1, 10) # Example: input 1 feature, output 10 features
        self.channel_linear = torch.nn.Linear(3 * 224 * 224, 10) # Example: 3 channels, 224x224
        self.sss_linear = torch.nn.Linear(3 * 224 * 224, 10) # Example: 3 channels, 224x224
        self.output_linear = torch.nn.Linear(30, 2) # Example: combined 3*10 features, output 2 classes
        self.num_classes = 2 # Crucial for confusion matrix

    # It MUST accept the same number of arguments as passed in train_multimodal_model and evaluate_multimodal_model
    def forward(self, main_image, channel_patch, sss_patch):
        # Implement a dummy forward pass that uses all inputs
        main_out = self.main_linear(main_image.view(main_image.size(0), -1))
        channel_out = self.channel_linear(channel_patch.view(channel_patch.size(0), -1))
        sss_out = self.sss_linear(sss_patch.view(sss_patch.size(0), -1))

        # Concatenate outputs (example simple fusion)
        combined_out = torch.cat((main_out, channel_out, sss_out), dim=1)
        return self.output_linear(combined_out)

class DummyDataset(Dataset):
    def __len__(self):
        return 3  # just a few batches

    def __getitem__(self, idx):
        return {
            "main_image": torch.randn(3, 224, 224),
            "label": torch.tensor(1),
            "channel_image": torch.randn(3, 224, 224),
            "sss_image": torch.randn(3, 224, 224),
            "patch_channels": {
                "patch_15_channel": torch.randn(3, 224, 224),
                "patch_30_channel": torch.randn(3, 224, 224)
            },
            "patch_sss": {
                "patch_15_sss": torch.randn(3, 224, 224),
                "patch_30_sss": torch.randn(3, 224, 224)
            }
        }

def get_dummy_loader():
    return DataLoader(DummyDataset(), batch_size=1)


def dummy_train_unimodal_model(**kwargs):
    return 0.9, 0.1

def dummy_evaluate_unimodal_model(**kwargs):
    return 0.85

def dummy_train_multimodal_model(**kwargs):
    return 0.8, 0.75

def dummy_multimodal_evaluate_model(**kwargs):
    return 0.78


class TestCheckpointing(unittest.TestCase):
    def test_save_model_and_load(self):
        model = DummyModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "dummy.csv")
            open(csv_path, "a").close()
            patch_type = "test"
            save_model(model, csv_path, patch_type)
        
            # Match the logic in save_model: go one level up
            base_path = os.path.dirname(os.path.dirname(csv_path))
            model_path = os.path.join(base_path, "models", f"bayesian_model_type{patch_type}.pth")
        
            self.assertTrue(os.path.exists(model_path)) 
            model2 = DummyModel()
            load_and_fix_state_dict(model2, model_path, device="cpu")
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                self.assertTrue(torch.equal(p1, p2))
class TestUnimodalTrainingEvaluation(unittest.TestCase):
    @patch("Multimodal_AUV.train.unimodal.train_unimodal_model")
    def test_train_and_evaluate_unimodal(self, mock_eval):
        model = DummyModel()
        writer = SummaryWriter(log_dir=tempfile.mkdtemp())
        loader = get_dummy_loader()

        # Do not patch `train_unimodal_model` — you want to test the real one now
        acc, loss = train_unimodal_model(
            model=model,
            dataloader=loader,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
            epoch=0,
            total_num_epochs=3,
            num_mc=1,
            sum_writer=writer,
            device="cpu",
            model_type="image",
            csv_path=os.path.join(tempfile.mkdtemp(), "log.csv")
        )

        self.assertIsInstance(acc, float)
        self.assertIsInstance(loss, float)

class TestOptimizersSchedulers(unittest.TestCase):
    def test_define_optimizers_and_schedulers(self):
        models_dict = {
            "image_model": DummyModel(),
            "channels_model": DummyModel(),
            "sss_model": DummyModel(),
            "multimodal_model": DummyModel(),
            "image_model_feat": DummyModel(),
            "channels_model_feat": DummyModel(),
            "sss_model_feat": DummyModel(),
        }
        optimizer_params = {k: {"lr": 0.001} for k in models_dict}
        scheduler_params = {k: {"step_size": 1, "gamma": 0.1} for k in optimizer_params}
        criterion, optimizers, schedulers = define_optimizers_and_schedulers(models_dict, optimizer_params, scheduler_params)
        self.assertIsInstance(criterion, nn.CrossEntropyLoss)
        for opt in optimizers.values():
            self.assertIsInstance(opt, optim.Optimizer)
        for sch in schedulers.values():
            self.assertIsInstance(sch, optim.lr_scheduler.StepLR)




import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from unittest.mock import patch
import tempfile
import os
from torch.utils.tensorboard import SummaryWriter

# Assuming these dummy models and loaders are defined elsewhere
# For completeness, let's include dummy implementations if not already present in your actual code
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2) # Example: input 10 features, output 2 classes
        self.num_classes = 2 # Required for metrics

    def forward(self, x):
        # Simplistic forward pass
        return self.linear(x)
class DummyModel_uni(nn.Module):
    def __init__(self):
        super().__init__()
        # Assuming the flattened input will be 224 features (from 672x224, maybe it's 3*224 for 3 images of 224 features each?)
        # Or if 224 is the number of features per sample, it should be 224
        # Let's assume input_features is the last dimension of the input tensor
        self.linear = nn.Linear(10, 2) # Adjust based on the actual input shape

    def forward(self, x):
        # If x is coming in as (batch_size, height, width), flatten it
        # Example for (N, H, W) -> (N, H*W)
        # x = x.view(x.size(0), -1)
        return self.linear(x)
class DummyDataset(Dataset):
    def __getitem__(self, idx):
        return torch.randn(10), torch.tensor(0, dtype=torch.long) # Features, Label

    def __len__(self):
        return 5 # Small dataset for testing

def get_dummy_loader():
    return DataLoader(DummyDataset(), batch_size=1)
class DummyDataset_uni(Dataset):
    def __getitem__(self, idx):
        return {
            "main_image": torch.randn(10),
            "label": torch.tensor(1, dtype=torch.long),
            "channel_image": torch.randn(10),
            "sss_image": torch.randn(10),
            "patch_channels": {"patch_30_channel": torch.randn(10)},
            "patch_sss": {"patch_30_sss": torch.randn(10)}
        }
    def __len__(self):
        return 3
def get_dummy_loader_uni():
    return DataLoader(DummyDataset_uni(), batch_size=1)
# You need this dummy function to act as the side_effect for the patch
def dummy_get_kl_loss(model):
    return torch.tensor(0.1) # Returns a fixed KL loss for testing

class TestUnimodalTrainingEvaluation(unittest.TestCase):
    # We only patch 'get_kl_loss' because it's an external dependency
    # we want to control, not the primary functions under test.
    @patch("Multimodal_AUV.train.unimodal.get_kl_loss", side_effect=dummy_get_kl_loss)
    def test_train_and_evaluate_unimodal(self, mock_kl_loss):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = DummyModel_uni()
            writer = SummaryWriter(log_dir=tmpdir)

            # Store the initial state of the model's parameters.
            # We'll compare this later to ensure training occurred.
            initial_model_state = {k: v.clone() for k, v in model.state_dict().items()}

            # Call the main function that orchestrates training and evaluation.
            # We are now testing this function directly, not its sub-components.
            train_and_evaluate_unimodal_model(
                model=model,
                train_loader=get_dummy_loader_uni(),
                test_loader=get_dummy_loader_uni(),
                criterion=nn.CrossEntropyLoss(),
                optimizer=optim.Adam(model.parameters(), lr=0.001),
                scheduler=optim.lr_scheduler.StepLR(optim.Adam(model.parameters(), lr=0.001), step_size=1),
                num_epochs=3,
                device="cpu",
                model_name="image",
                save_dir=tmpdir,
                num_mc=2, # Set to 2 or more to avoid the var() warning
                sum_writer=writer
            )

            # --- Assertions on Side Effects ---

            # 1. Verify that the model's weights have changed.
            # This confirms that the training process actually updated the model.
            final_model_state = model.state_dict()
            weights_changed = False
            for k in initial_model_state:
                if not torch.equal(initial_model_state[k], final_model_state[k]):
                    weights_changed = True
                    break
            self.assertTrue(weights_changed, "Model parameters should have changed after training.")

          
            # 3. Confirm that TensorBoard log files were generated.
            # This indicates that logging functionality is active.
            tensorboard_log_dir = tmpdir
            self.assertTrue(os.path.exists(tensorboard_log_dir), "TensorBoard log directory was not created.")
            event_files = [f for f in os.listdir(tensorboard_log_dir) if f.startswith("events.out.tfevents")]
            self.assertGreater(len(event_files), 0, "No TensorBoard event files were found in the log directory.")

            # 4. Assert that the get_kl_loss mock was called (since it's still patched).
            mock_kl_loss.assert_called()


# Dummy get_kl_loss
def dummy_get_kl_loss(model):
    # Ensure it always returns a tensor
    return torch.tensor(0.1) # Or 0.0 if you prefer a neutral KL in dummy

# Dummy train_multimodal_model (this needs to be comprehensive enough to avoid errors)
def dummy_train_multimodal_model(
    multimodal_model, train_loader, criterion, optimizer, epoch,
    total_num_epochs, device, model_type, channel_patch_type, sss_patch_type,
    **kwargs # Accept other kwargs if needed by the real function
):
    logging.info("dummy_train_multimodal_model called")
    multimodal_model.train() # Set model to train mode for dropout
    total_loss = 0
    kl_losses = [] # Initialize kl_losses
    
    # Simulate processing a few batches
    for i, batch in enumerate(train_loader):
        if i >= 2: # Limit batches processed to speed up dummy
            break
        
        inputs = batch["main_image"].to(device)
        labels = batch["label"].long().to(device)
        channels = batch["channel_image"].to(device)
        sss = batch["sss_image"].to(device)

        # For dummy, simplify patch selection, just pass the main images
        # The real function's patch logic needs to be robust, but for dummy:
        channel_patch = batch["patch_channels"].get(channel_patch_type, channels)
        sss_patch = batch["patch_sss"].get(sss_patch_type, sss)

        optimizer.zero_grad()
        outputs = multimodal_model(inputs, channel_patch, sss_patch) # Use dummy values for actual patch types
        
        # Calculate KL loss (using the dummy get_kl_loss)
        kl = dummy_get_kl_loss(multimodal_model) # CALL THE DUMMY KL LOSS HERE
        kl_losses.append(kl)
        
        loss_ce = criterion(outputs, labels)
        
        # Simulate KL scaling as in actual train_multimodal_model
        kl_weight = (2 ** (epoch + 1)) / (2 ** total_num_epochs)
        scaled_kl = torch.mean(torch.stack(kl_losses), dim=0) / train_loader.batch_size * kl_weight
        
        loss = loss_ce + scaled_kl
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(train_loader) # Example loss

# Dummy evaluate_multimodal_model
def dummy_multimodal_evaluate_model(
    multimodal_model, dataloader, device, epoch, total_num_epochs, num_mc,
    model_type, channel_patch_type, sss_patch_type, csv_path, **kwargs
):
    logging.info("dummy_multimodal_evaluate_model called")
    multimodal_model.eval() # Set model to eval mode
    total_loss = 0
    correct = 0
    total = 0
    all_predicted = []
    all_labels = []
    all_predictive_uncertainty = []
    all_model_uncertainty = []
    kl_mc = [] # Initialize kl_mc

    criterion = nn.CrossEntropyLoss()
    epsilon = 1e-8 # for numerical stability

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 2: # Limit batches processed
                break
            
            inputs = batch["main_image"].to(device)
            labels = batch["label"].long().to(device)
            channels = batch["channel_image"].to(device)
            sss = batch["sss_image"].to(device)

            channel_patch = batch["patch_channels"].get(channel_patch_type, channels)
            sss_patch = batch["patch_sss"].get(sss_patch_type, sss)

            outputs_mc = []
            softmax_outputs_mc = []
            
            for _ in range(num_mc):
                outputs = multimodal_model(inputs, channel_patch, sss_patch)
                outputs_mc.append(outputs)
                softmax_outputs_mc.append(F.softmax(outputs, dim=1))
                
                kl = dummy_get_kl_loss(multimodal_model) # CALL THE DUMMY KL LOSS HERE
                kl_mc.append(kl)

            outputs_stack = torch.stack(outputs_mc)
            softmax_stack = torch.stack(softmax_outputs_mc)
            
            output_mean = torch.mean(outputs_stack, dim=0)
            if output_mean.ndim == 3 and output_mean.size(1) == 1: # Squeeze if needed
                output_mean = output_mean.squeeze(1)

            # Simulate KL scaling (assuming evaluate also calculates it for logging)
            kl_weight = (2 ** (epoch + 1)) / (2 ** total_num_epochs)
            kl_mean = torch.mean(torch.stack(kl_mc), dim=0) / len(dataloader) # Ensure kl_mc is not empty
            kl_scaled = kl_mean * kl_weight
            
            cross_entropy_loss = criterion(output_mean, labels)
            loss = cross_entropy_loss + kl_scaled # Include KL in dummy loss if real does
            total_loss += loss.item()

            _, predicted = torch.max(output_mean, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Dummy uncertainty calcs
            mean_softmax = torch.mean(softmax_stack, dim=0)
            predictive_uncertainty_batch = -torch.sum(mean_softmax * torch.log(mean_softmax + epsilon), dim=1)
            all_predictive_uncertainty.extend(predictive_uncertainty_batch.cpu().detach().numpy())
            
            # Dummy model uncertainty
            entropy_per_mc_sample = -torch.sum(softmax_stack * torch.log(softmax_stack + epsilon), dim=2)
            aleatoric_uncertainty_batch = torch.mean(entropy_per_mc_sample, dim=0)
            model_uncertainty_batch = predictive_uncertainty_batch - aleatoric_uncertainty_batch
            all_model_uncertainty.extend(model_uncertainty_batch.cpu().numpy())
            
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_accuracy = correct / total if total > 0 else 0.0
    test_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    
    # Write to CSV (dummy, as csv.writer is mocked in test_multimodal_evaluate_model_runs)
    # But for test_train_and_evaluate_multimodal_2, csv.writer is NOT mocked, so it writes to file
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists:
            csv_writer.writerow([
                "Epoch", "Model Type", "Test Loss", "Test Accuracy",
                "Predictive Uncertainty", "Model Uncertainty",
                "Scaled KL", "Cross Entropy Loss",
                "Channel Patch Type", "SSS Patch Type"
            ])
        csv_writer.writerow([
            epoch + 1, model_type, test_loss, test_accuracy,
            np.mean(all_predictive_uncertainty), np.mean(all_model_uncertainty),
            kl_scaled.item(), cross_entropy_loss.item(),
            channel_patch_type, sss_patch_type
        ])
    
    return test_accuracy

# Dummy Dataset (as you provided for this test)
class DummyDataset_2(Dataset):
    def __len__(self):
        return 3

    def __getitem__(self, idx):
        label = 0 if idx % 2 == 0 else 1
        return {
            "main_image": torch.randn(3, 224, 224),
            "label": torch.tensor(label).long(),
            "channel_image": torch.randn(3, 224, 224),
            "sss_image": torch.randn(3, 224, 224),
            "patch_channels": {
                "patch_15_channel": torch.randn(3, 224, 224),
                "patch_30_channel": torch.randn(3, 224, 224)
            },
            "patch_sss": {
                "patch_15_sss": torch.randn(3, 224, 224),
                "patch_30_sss": torch.randn(3, 224, 224)
            }
        }
import logging # Import logging

# Configure logging for clearer output in tests
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

# Dummy get_dummy_loader (as you provided)
def get_dummy_loader():
    return DataLoader(DummyDataset_2(), batch_size=1)


# Dummy Multimodal Model
# Dummy Multimodal Model - MODIFIED to expect 1 feature input

class com_DummyBayesianModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Model now expects image input (3, 224, 224)
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Reduces 224 to 112
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Reduces 112 to 56
        )
        # Calculate the flattened size: 32 channels * 56 * 56
        self.flattened_size = 32 * 56 * 56
        self.linear = nn.Linear(self.flattened_size, 2) # Output for 2 classes
        self.num_classes = 2

    def forward(self, main_image, channel_image=None, sss_image=None, patch_channels=None, patch_sss=None):
        # CORRECTED: Pass main_image through convolutional features, then flatten.
        x = self.features(main_image)
        x = torch.flatten(x, 1) # Flatten starting from dimension 1 (batch dimension)
        return self.linear(x) # Now `x` is the flattened output, suitable for `self.linear`

# Dummy Multimodal Dataset (identical to DummyDataset_uni from your previous example)

# Dummy Multimodal Dataset - Now matches your provided `DummyDataset_2`
class DummyDataset_2(Dataset):
    def __len__(self):
        return 3

    def __getitem__(self, idx):
        label = 0 if idx % 2 == 0 else 1
        return {
            "main_image": torch.randn(3, 224, 224),
            "label": torch.tensor(label).long(), # Labels are long (class indices), which is standard for CrossEntropyLoss
            "channel_image": torch.randn(3, 224, 224),
            "sss_image": torch.randn(3, 224, 224),
            "patch_channels": {
                "patch_15_channel": torch.randn(3, 224, 224),
                "patch_30_channel": torch.randn(3, 224, 224)
            },
            "patch_sss": {
                "patch_15_sss": torch.randn(3, 224, 224),
                "patch_30_sss": torch.randn(3, 224, 224)
            }
        }

def get_dummy_loader_multimodal(): # Renamed to avoid clash with get_dummy_loader_uni
    return DataLoader(DummyDataset_2(), batch_size=1) # Use the new DummyDataset_2

def dummy_get_kl_loss(model):
    return torch.tensor(0.1)
# Dummy implementations for the actual training/evaluation functions if they are needed for side_effect
# However, for this test, we want to run the *real* functions, so these won't be used as side_effects
# for train_multimodal_model and evaluate_multimodal_model.
# If these functions themselves are under test, you'd test them directly in separate unit tests.

# --- Import the actual functions to be tested ---
# Assuming these are imported by your train_and_evaluate_multimodal_model function
from Multimodal_AUV.train.multimodal import train_multimodal_model, evaluate_multimodal_model
# And the orchestrating function
from Multimodal_AUV.train.loop_utils import train_and_evaluate_multimodal_model # Assuming this is the top-level func

# Ensure kl_loss is available if used
# Assuming it's imported from Multimodal_AUV.train.multimodal or a utility file
def dummy_get_kl_loss(model):
    return torch.tensor(0.1)


# --- The Test Class ---
class TestMultimodalTrainingEvaluation(unittest.TestCase):
    @patch("Multimodal_AUV.train.multimodal.get_kl_loss", side_effect=dummy_get_kl_loss)
    def test_train_and_evaluate_multimodal_2(self, mock_kl_loss):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = com_DummyBayesianModel() # This now uses the modified dummy model
            train_loader_instance = get_dummy_loader_multimodal()
            test_loader_instance = get_dummy_loader_multimodal()
            writer = SummaryWriter(log_dir=tmpdir)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)

            initial_model_state = {k: v.clone() for k, v in model.state_dict().items()}

            train_and_evaluate_multimodal_model(
                train_loader=train_loader_instance,
                test_loader=test_loader_instance,
                multimodal_model=model,
                criterion=nn.CrossEntropyLoss(),
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                num_epochs=1,
                num_mc=2,
                device="cpu",
                model_type="multimodal",
                # Make sure these patch types exist in your DummyDataset_2
                # Your DummyDataset_2 provides "patch_15_channel", "patch_30_channel" etc.
                # Adjust these to match the keys your dataset provides if you want to use them.
                # Otherwise, the .get() in train_multimodal_model will fallback to full tensor.
                channel_patch_type="patch_30_channel", # Using one of the new patch types
                sss_patch_type="patch_30_sss",       # Using one of the new patch types
                csv_path=os.path.join(tmpdir, "results.csv"),
                sum_writer=writer
            )

            final_model_state = model.state_dict()
            weights_changed = False
            for k in initial_model_state:
                if not torch.equal(initial_model_state[k], final_model_state[k]):
                    weights_changed = True
                    break
            self.assertTrue(weights_changed, "Multimodal model parameters should have changed after training.")

           
            tensorboard_log_dir = tmpdir
            self.assertTrue(os.path.exists(tensorboard_log_dir), "TensorBoard log directory was not created.")
            event_files = [f for f in os.listdir(tensorboard_log_dir) if f.startswith("events.out.tfevents")]
            self.assertGreater(len(event_files), 0, "No TensorBoard event files found for multimodal training.")

            results_csv_path = os.path.join(tmpdir, "results.csv")
            self.assertTrue(os.path.exists(results_csv_path),
                            f"Results CSV file expected at {results_csv_path} should exist.")
            with open(results_csv_path, 'r') as f:
                lines = f.readlines()
                self.assertGreater(len(lines), 1, "Results CSV should have at least a header and one data row.")

            mock_kl_loss.assert_called()
class DummyMultimodalDataset(Dataset):
    def __getitem__(self, idx):
        return {
            "main_image": torch.randn(10),          # shape: (features=10)
            "label": torch.tensor(1, dtype=torch.long),  # scalar label
            "channel_image": torch.randn(10),      # shape: (10,)
            "sss_image": torch.randn(10),          # shape: (10,)
            "patch_channels": {"patch_30_channel": torch.randn(10)},
            "patch_sss": {"patch_30_sss": torch.randn(10)}
        }


    def __len__(self):
        return 3


def dummy_get_kl_loss(model):
    return torch.tensor(0.1)

def get_dummy_optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=0.01)

class DummyBayesianModel_1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 2) # Example: input 1 feature, output 2 classes
        self.num_classes = 2 # Add num_classes attribute for the confusion matrix
    def forward(self, main_image, channel_patch, sss_patch):
        # Simplistic forward pass for testing
        return self.linear(main_image)

class TestBayesianMultimodalTraining(unittest.TestCase):
    @patch("Multimodal_AUV.train.multimodal.get_kl_loss", side_effect=dummy_get_kl_loss)
    @patch("Multimodal_AUV.train.checkpointing.save_model")
    def test_train_multimodal_model(self, mock_save_model, mock_kl):
        model = DummyBayesianModel_1()
        dataset = DummyMultimodalDataset()
        dataloader = DataLoader(dataset, batch_size=1)
        criterion = nn.CrossEntropyLoss()
        optimizer = get_dummy_optimizer(model)
        device = torch.device("cpu")
        model_type = "bayesian_test"
        epoch = 0
        total_num_epochs = 1
        num_mc = 1

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "log.csv")
            loss, acc = train_multimodal_model(
                multimodal_model=model,
                dataloader=dataloader,
                criterion=criterion,
                optimizer=optimizer,
                epoch=epoch,
                device=device,
                model_type=model_type,
                total_num_epochs=total_num_epochs,
                num_mc=num_mc,
                sum_writer=lambda *a, **k: None,
                channel_patch_type="patch_30_channel",
                sss_patch_type="patch_30_sss",
                csv_path=csv_path
            )
            self.assertIsInstance(loss, float)
            self.assertIsInstance(acc, float)
            self.assertTrue(os.path.exists(csv_path))
            with open(csv_path, "r") as f:
                lines = list(csv.reader(f))
            self.assertGreaterEqual(len(lines), 1)


# Dummy implementations for testing
class DummyBayesianModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 2) # Example: input 1 feature, output 2 classes
        self.num_classes = 2 # Add num_classes attribute for the confusion matrix
    def forward(self, main_image, channel_patch, sss_patch):
        # Simplistic forward pass for testing
        return self.linear(main_image)

class DummyMultimodalDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 5

    def __getitem__(self, idx):
        label = 0 if idx % 2 == 0 else 1
        # Ensure label is a scalar tensor for CrossEntropyLoss target
        return {
            "main_image": torch.randn(1, 1), # This needs to match input for linear(1,2)
            "label": torch.tensor(label).long(), # Explicitly .long() here too
            "channel_image": torch.randn(3, 224, 224),
            "sss_image": torch.randn(3, 224, 224),
            "patch_channels": {"patch_30_channel": torch.randn(3, 224, 224)},
            "patch_sss": {"patch_30_sss": torch.randn(3, 224, 224)},
        }

def dummy_get_kl_loss(model):
    return torch.tensor(0.1) # Dummy KL loss




class TestMultimodalEvaluateModel(unittest.TestCase):
    @patch("Multimodal_AUV.train.multimodal.get_kl_loss", side_effect=dummy_get_kl_loss)
    @patch("os.path.isfile", return_value=False)
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.subplots", return_value=(MagicMock(), MagicMock()))
    @patch("matplotlib.pyplot.close")
    @patch("matplotlib.pyplot.title")
    @patch("sklearn.metrics.ConfusionMatrixDisplay.plot")
    def test_multimodal_evaluate_model_runs(self, mock_get_kl_loss, mock_isfile, mock_savefig, mock_subplots, mock_close, mock_title, mock_disp_plot):
        # Now the arguments match the patch order (from bottom up)

        model = DummyBayesianModel()
        dataset = DummyMultimodalDataset()
        dataloader = DataLoader(dataset, batch_size=1)
        device = torch.device("cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test_metrics.csv")
            accuracy = evaluate_multimodal_model(
                multimodal_model=model,
                dataloader=dataloader,
                device=device,
                epoch=0,
                total_num_epochs=1,
                num_mc=1,
                model_type="test_model",
                channel_patch_type="patch_30_channel",
                sss_patch_type="patch_30_sss",
                csv_path=csv_path,
            )

            self.assertIsInstance(accuracy, float)
            self.assertTrue(os.path.exists(csv_path))
            with open(csv_path, "r") as f:
                lines = list(csv.reader(f))
            self.assertGreaterEqual(len(lines), 2)

            # Assert calls on the correctly named mocks
            mock_savefig.assert_called_once()
            mock_close.assert_called_once()
            # If you need to assert on get_kl_loss:
            mock_get_kl_loss.assert_called()
class DummyDropoutModel(nn.Module):
    def __init__(self, output_dim=2):
        super().__init__()
        self.linear = nn.Linear(10, output_dim)
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        return self.linear(self.dropout(x))

class DummyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 4
    def __getitem__(self, idx):
        return {
            "main_image": torch.rand(10),
            "label": torch.tensor(1, dtype=torch.long),  # scalar label
            "channel_image": torch.rand(10),
            "sss_image": torch.rand(10),
            "patch_channels": {},
            "patch_sss": {}
        }


class TestUnimodalEvaluateModel(unittest.TestCase):
    @patch("bayesian_torch.models.dnn_to_bnn.get_kl_loss", side_effect=mock_kl_loss)
    @patch("os.path.isfile", return_value=False)
    @patch("csv.writer")
    @patch("builtins.open", new_callable=mock_open)
    def test_evaluate_unimodal_model(self, mock_open_file, mock_csv_writer, mock_file_exists, mock_kl_loss_func):
        model = DummyDropoutModel()
        dataloader = DataLoader(DummyDataset(), batch_size=2)
        device = torch.device("cpu")
        csv_path = "dummy_path.csv"

        mock_writer_instance = MagicMock()
        mock_csv_writer.return_value = mock_writer_instance

        accuracy = evaluate_unimodal_model(
            model=model,
            dataloader=dataloader,
            device=device,
            epoch=0,
            csv_path=csv_path,
            total_num_epochs=10,
            num_mc=3,
            model_type="image"
        )

        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
        mock_open_file.assert_called_once_with(csv_path, mode='a', newline='')
        mock_writer_instance.writerow.assert_called()
