import os
import csv
import tempfile
import unittest
from unittest import mock, patch, MagicMock

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
import pytest

from Multimodal_AUV.models.model_utils import save_model,load_and_fix_state_dict,define_optimizers_and_schedulers,train_and_evaluate_unimodal_model,train_and_evaluate_multimodal_model,multimodal_evaluate_model,train_multimodal_model,evaluate_unimodal_model

# Dummy model for testing
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

# Dummy DataLoader
class DummyDataLoader:
    def __init__(self, batch_size=4):
        self.batch_size = batch_size
    def __iter__(self):
        return iter([ (torch.randn(4, 10), torch.randint(0, 2, (4,))) ])
    def __len__(self):
        return 1

# Dummy train/eval functions to patch
def dummy_train_unimodal_model(**kwargs):
    return 0.9, 0.1

def dummy_evaluate_unimodal_model(**kwargs):
    return 0.85

def dummy_train_multimodal_model(**kwargs):
    return 0.8, 0.75

def dummy_multimodal_evaluate_model(**kwargs):
    return 0.78

# Define test functions
def test_save_model_and_load():
    model = DummyModel()
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "dummy.csv")
        open(csv_path, "a").close()
        patch_type = "test"
        save_model(model, csv_path, patch_type)
        model_path = os.path.join(tmpdir, "models", f"bayesian_model_type{patch_type}.pth")
        assert os.path.exists(model_path)
        model2 = DummyModel()
        load_and_fix_state_dict(model2, model_path, device="cpu")
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)

def test_define_optimizers_and_schedulers():
    models_dict = {
        "image_model": DummyModel(),
        "channels_model": DummyModel(),
        "sss_model": DummyModel(),
        "multimodal_model": DummyModel(),
        "image_model_feat": DummyModel(),
        "channels_model_feat": DummyModel(),
        "sss_model_feat": DummyModel(),
    }
    optimizer_params = {k.split('_')[0]: {"lr": 0.001} for k in models_dict}
    scheduler_params = {k: {"step_size": 1, "gamma": 0.1} for k in optimizer_params}
    criterion, optimizers, schedulers = define_optimizers_and_schedulers(models_dict, optimizer_params, scheduler_params)
    assert isinstance(criterion, nn.CrossEntropyLoss)
    for opt in optimizers.values():
        assert isinstance(opt, optim.Optimizer)
    for sch in schedulers.values():
        assert isinstance(sch, optim.lr_scheduler._LRScheduler)

def test_train_and_evaluate_unimodal():
    with tempfile.TemporaryDirectory() as tmpdir:
        model = DummyModel()
        writer = SummaryWriter(log_dir=tmpdir)
        with mock.patch("ml_module.utils.model_utils.train_unimodal_model", side_effect=dummy_train_unimodal_model), \
             mock.patch("ml_module.utils.model_utils.evaluate_unimodal_model", side_effect=dummy_evaluate_unimodal_model):
             train_and_evaluate_unimodal_model(
                model=model,
                train_loader=DummyDataLoader(),
                test_loader=DummyDataLoader(),
                criterion=nn.CrossEntropyLoss(),
                optimizer=optim.Adam(model.parameters(), lr=0.001),
                scheduler=optim.lr_scheduler.StepLR(optim.Adam(model.parameters(), lr=0.001), step_size=1),
                num_epochs=1,
                device="cpu",
                model_name="unimodal_test",
                save_dir=tmpdir,
                num_mc=1,
                sum_writer=writer
            )

def test_train_and_evaluate_multimodal():
    with tempfile.TemporaryDirectory() as tmpdir:
        model = DummyModel()
        writer = SummaryWriter(log_dir=tmpdir)
        with mock.patch("ml_module.utils.model_utils.train_multimodal_model", side_effect=dummy_train_multimodal_model), \
             mock.patch("ml_module.utils.model_utils.multimodal_evaluate_model", side_effect=dummy_multimodal_evaluate_model):
            train_and_evaluate_multimodal_model(
                train_loader=DummyDataLoader(),
                test_loader=DummyDataLoader(),
                multimodal_model=model,
                criterion=nn.CrossEntropyLoss(),
                optimizer=optim.Adam(model.parameters(), lr=0.001),
                lr_scheduler=optim.lr_scheduler.StepLR(optim.Adam(model.parameters(), lr=0.001), step_size=1),
                num_epochs=1,
                num_mc=1,
                device="cpu",
                model_type="multimodal_test",
                channel_patch_type="cpatch",
                sss_patch_type="spatch",
                csv_path=os.path.join(tmpdir, "results.csv"),
                sum_writer=writer
            )


# Dummy Bayesian model for test
class DummyBayesianModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)
    def forward(self, x1, x2, x3):
        return self.linear(x1)

# Dummy dataset for multimodal inputs
class DummyMultimodalDataset(Dataset):
    def __getitem__(self, idx):
        return {
            "main_image": torch.randn(10),
            "label": torch.tensor(1),
            "channel_image": torch.randn(10),
            "sss_image": torch.randn(10),
            "patch_channels": {"patch_30_channel": torch.randn(10)},
            "patch_sss": {"patch_30_sss": torch.randn(10)}
        }
    def __len__(self):
        return 3

# Mock SummaryWriter function
def mock_writer(tag, scalar_value, global_step):
    pass

# Dummy KL loss
def dummy_get_kl_loss(model):
    return torch.tensor(0.1)

# Dummy optimizer
def get_dummy_optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=0.01)

# Setup test for train_multimodal_model
def test_train_multimodal_model():
    model = DummyBayesianModel()
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
        with patch("ml_module.utils.model_utils.get_kl_loss", side_effect=dummy_get_kl_loss), \
             patch("ml_module.utils.model_utils.save_model") as mock_save_model:

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
                sum_writer=mock_writer,
                channel_patch_type="patch_30_channel",
                sss_patch_type="patch_30_sss",
                csv_path=csv_path
            )

            # Assertions
            assert isinstance(loss, float), "Loss should be a float"
            assert isinstance(acc, float), "Accuracy should be a float"
            assert os.path.exists(csv_path), "CSV log file should exist"
            with open(csv_path, "r") as f:
                lines = list(csv.reader(f))
                assert len(lines) >= 2, "CSV should have header and at least one row"




# Dummy model for testing
class DummyMultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    def forward(self, x1, x2, x3):
        return self.fc(x1)

# Dummy dataset
class DummyMultimodalDataset(Dataset):
    def __getitem__(self, idx):
        return {
            "main_image": torch.randn(1, 10),
            "label": torch.tensor(1),
            "channel_image": torch.randn(1, 10),
            "sss_image": torch.randn(1, 10),
            "patch_channels": {"patch_30_channel": torch.randn(1, 10)},
            "patch_sss": {"patch_30_sss": torch.randn(1, 10)}
        }
    def __len__(self):
        return 3

# Dummy get_kl_loss function
def dummy_get_kl_loss(model):
    return torch.tensor(0.1)

# Dummy confusion matrix function patch
def dummy_confusion_matrix(true, pred):
    return np.array([[1, 0], [0, 2]])

# Dummy ConfusionMatrixDisplay
class DummyConfusionMatrixDisplay:
    def __init__(self, confusion_matrix):
        self.confusion_matrix = confusion_matrix
    def plot(self, cmap=None, ax=None):
        pass

# Patch matplotlib functions
plt.subplots = lambda figsize: (MagicMock(), MagicMock())
plt.savefig = lambda path: None
plt.close = lambda fig: None

# Test class
class TestMultimodalEvaluateModel(unittest.TestCase):
    @patch("ml_module.utils.model_utils.get_kl_loss", side_effect=dummy_get_kl_loss)
    @patch("ml_module.utils.model_utils.confusion_matrix", side_effect=dummy_confusion_matrix)
    @patch("ml_module.utils.model_utils.ConfusionMatrixDisplay", side_effect=DummyConfusionMatrixDisplay)
    def test_multimodal_evaluate_model_runs(self, mock_disp, mock_conf, mock_kl):

        model = DummyMultimodalModel()
        dataloader = DataLoader(DummyMultimodalDataset(), batch_size=1)
        device = torch.device("cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test_metrics.csv")
            accuracy = multimodal_evaluate_model(
                multimodal_model=model,
                dataloader=dataloader,
                device=device,
                epoch=0,
                total_num_epochs=1,
                num_mc=1,
                model_type="test_model",
                channel_patch_type="patch_30_channel",
                sss_patch_type="patch_30_sss",
                csv_path=csv_path
            )

            self.assertIsInstance(accuracy, float)
            self.assertTrue(os.path.exists(csv_path))
            with open(csv_path, "r") as f:
                lines = list(csv.reader(f))
                self.assertGreaterEqual(len(lines), 2)

unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(TestMultimodalEvaluateModel))

# Dummy model with dropout
class DummyDropoutModel(nn.Module):
    def __init__(self, output_dim=2):
        super().__init__()
        self.linear = nn.Linear(10, output_dim)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        return self.linear(self.dropout(x))

# Dummy batch generator
class DummyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        data = {
            "main_image": torch.rand(10),
            "label": torch.tensor(1),
            "channel_image": torch.rand(10),
            "sss_image": torch.rand(10),
            "patch_channels": {},
            "patch_sss": {}
        }
        return data

def mock_kl_loss(model):
    return torch.tensor(0.05)

@patch("your_module_path.get_kl_loss", side_effect=mock_kl_loss)
@patch("your_module_path.os.path.isfile", return_value=False)
@patch("your_module_path.csv.writer")
@patch("your_module_path.open", new_callable=mock_open)
def test_evaluate_unimodal_model(mock_open_file, mock_csv_writer, mock_file_exists, mock_kl_loss_func):
    # Arrange
    model = DummyDropoutModel()
    dataloader = DataLoader(DummyDataset(), batch_size=2)
    device = torch.device("cpu")
    csv_path = "dummy_path.csv"

    # Patch the CSV writer object to catch rows written
    mock_writer_instance = MagicMock()
    mock_csv_writer.return_value = mock_writer_instance

    # Act
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

    # Assert
    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0
    mock_open_file.assert_called_once_with(csv_path, mode='a', newline='')
    mock_writer_instance.writerow.assert_called()  # Ensure it writes at least once
