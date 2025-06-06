import torch
import torch.nn as nn
from torchvision.models import resnet50
from types import SimpleNamespace
import tempfile
import os
from Multimodal_AUV.models.model_utils import define_models, load_models, load_pretrained_resnet_as_feature_extractor
from Multimodal_AUV.models.base_models import ResNet50Custom, AdditiveAttention, MultiModalModel
import unittest
from unittest import mock
import torch
import torch.nn as nn
from typing import Dict, Any
import os
from unittest import mock  
from unittest.mock import patch, MagicMock  
# Adjust the imports to match your project structure
from Multimodal_AUV.models.model_utils import define_models
from bayesian_torch.models.dnn_to_bnn import bnn_linear_layer, bnn_conv_layer, bnn_lstm_layer
class DummyFeatureExtractor(nn.Module):
    def __init__(self, output_size=128):
        super().__init__()
        self.output_size = output_size
    def forward(self, x):
        batch_size = x.shape[0]
        # Return dummy tensor of shape (batch_size, output_size)
        return torch.ones(batch_size, self.output_size)

class Test_Models(unittest.TestCase):
    def test_resnet50custom_output_shape(self):
        model = ResNet50Custom(input_channels=3, num_classes=10)
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        assert output.shape == (1, 10)


    def test_additive_attention_output_shape(self):
        attention = AdditiveAttention(d_model=2048, hidden_dim=128)
        dummy_input = torch.randn(1, 2048)
        output = attention(dummy_input)
        assert output.shape == (1, 128)


    def test_multimodal_model_forward_pass(self):
        dummy_feat = DummyFeatureExtractor(output_size=128)
        model = MultiModalModel(dummy_feat, dummy_feat, dummy_feat, num_classes=5)

        x = torch.randn(2, 3, 224, 224)  # image
        channels = torch.randn(2, 3, 224, 224)
        sss = torch.randn(2, 3, 224, 224)

        # Override attention modules to bypass attention math in this unit test
        model.attention_image = nn.Identity()
        model.attention_channels = nn.Identity()
        model.attention_sss = nn.Identity()

        out = model(x, channels, sss)
        assert out.shape == (2, 5)


    def test_load_pretrained_resnet_as_feature_extractor_default(self):
        model = load_pretrained_resnet_as_feature_extractor()
        assert isinstance(model, nn.Module)
        assert isinstance(model.fc, nn.Module)  # Should be Identity


    def test_load_pretrained_resnet_as_feature_extractor_grayscale(self):
        model = load_pretrained_resnet_as_feature_extractor(input_channels=1)
        assert isinstance(model, nn.Module)
        assert model.conv1.in_channels == 1


    def test_load_models(self):
            with tempfile.TemporaryDirectory() as tmpdirname:
                dummy_model = resnet50(weights=None)
                dummy_model.fc = nn.Identity()
                dummy_path = os.path.join(tmpdirname, "dummy_model.pth")
                torch.save(dummy_model.state_dict(), dummy_path)

                model_paths = {
                    "image": dummy_path,
                    "channels": dummy_path,
                    "sss": dummy_path
                }

                device = torch.device("cpu")
                img_model, chan_model, sss_model = load_models(model_paths, device, num_classes=10)

                self.assertIsInstance(img_model, nn.Module)
                self.assertIsInstance(chan_model, nn.Module)
                self.assertIsInstance(sss_model, nn.Module)


    @mock.patch("Multimodal_AUV.models.model_utils.dnn_to_bnn")
    def test_define_models_with_mocked_bnn(self, mock_dnn_to_bnn):
        # Mock the dnn_to_bnn function to do nothing
        mock_dnn_to_bnn.side_effect = lambda model, const_bnn_prior_parameters: None

        model_paths = {
            "image": None,
            "channels": None,
            "sss": None
        }
        device = torch.device("cpu")
        const_bnn_prior_parameters = {}

        models = define_models(model_paths, device, num_classes=3, const_bnn_prior_parameters=const_bnn_prior_parameters)

        self.assertIsInstance(models["image_model"], ResNet50Custom)
        self.assertIsInstance(models["channels_model"], ResNet50Custom)
        self.assertIsInstance(models["sss_model"], ResNet50Custom)
        self.assertIsInstance(models["multimodal_model"], nn.Module)


class Test_DefineModelsBNN(unittest.TestCase):
        def setUp(self):
            self.device = torch.device("cpu")
            self.num_classes = 3
            self.const_bnn_prior_parameters = {
              
                    "prior_mu": 0.0,
                    "prior_sigma": 1.0,
                    "posterior_mu_init": 0.0,
                    "posterior_rho_init": -3.0,
                    "type": "Reparameterization",
                    "moped_enable": True,
                    "moped_delta": 0.1,
                }
         
            self.model_paths = {
                "image": "dummy_image.pth",
                "channels": "dummy_channels.pth",
                "sss": "dummy_sss.pth",
            }

        def is_bayesian_model(self, model: nn.Module) -> bool:
            """Check that all Conv, Linear, and LSTM layers have been replaced by Bayesian versions."""
            for layer in model.modules():
                # Disallow any unconverted nn.Linear, nn.Conv*, or nn.LSTM layers
                if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.LSTM)):
                    return False
                # Optionally: check that Bayesian layers implement kl_loss
                if isinstance(layer, (bnn_linear_layer().__class__, bnn_conv_layer().__class__, bnn_lstm_layer().__class__)):
                    if not hasattr(layer, "kl_loss"):
                        return False
            return True


        @mock.patch("Multimodal_AUV.models.model_utils.load_models")
        def test_define_models_returns_all_bayesian(self, mock_load_models):
            dummy_feat = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=3),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(8 * 6 * 6, self.num_classes),
            )
            mock_load_models.return_value = (dummy_feat, dummy_feat, dummy_feat)

            models = define_models(
                model_paths=self.model_paths,
                device=self.device,
                num_classes=self.num_classes,
                const_bnn_prior_parameters=self.const_bnn_prior_parameters,
            )

            for name, model in models.items():
                with self.subTest(model_name=name):
                    self.assertTrue(self.is_bayesian_model(model), f"{name} is not fully Bayesian")

