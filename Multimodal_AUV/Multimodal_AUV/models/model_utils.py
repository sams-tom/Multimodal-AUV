import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from typing import Dict, Any, Tuple
import logging
from bayesian_torch import dnn_to_bnn
import os
from base_models import ResNet50Custom, MultiModalModel, Identity

def define_models(
    model_paths: Dict[str, str],
    device: torch.device,
    num_classes: int,
    const_bnn_prior_parameters: Dict[str, Any]
) -> Dict[str, nn.Module]:
    """
    Defines and initializes unimodal and multimodal models, converts specified models to Bayesian Neural Networks.
    """
    try:
        image_model = ResNet50Custom(input_channels=3, num_classes=num_classes)
        channels_model = ResNet50Custom(input_channels=3, num_classes=num_classes)
        sss_model = ResNet50Custom(input_channels=1, num_classes=num_classes)

        logging.info("Loading pretrained models as feature extractors.")

        image_model_feat, channels_model_feat, sss_model_feat = load_models(
            model_paths, device, num_classes
        )
        multimodal_model = MultiModalModel(image_model_feat, channels_model_feat, sss_model_feat, num_classes)

        logging.info("Converting models to Bayesian versions.")
        dnn_to_bnn(image_model, const_bnn_prior_parameters)
        dnn_to_bnn(channels_model, const_bnn_prior_parameters)
        dnn_to_bnn(sss_model, const_bnn_prior_parameters)
        dnn_to_bnn(multimodal_model, const_bnn_prior_parameters)

        return {
            "image_model": image_model,
            "channels_model": channels_model,
            "sss_model": sss_model,
            "multimodal_model": multimodal_model,
            "image_model_feat": image_model_feat,
            "channels_model_feat": channels_model_feat,
            "sss_model_feat": sss_model_feat,
        }

    except Exception as e:
        logging.error(f"Error defining models: {e}", exc_info=True)
        raise

    
def load_pretrained_resnet_as_feature_extractor(input_channels: int = 3) -> nn.Module:
    """
    Loads a pretrained ResNet50 model and adapts the input convolutional layer if input_channels != 3.
    """
    try:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        if input_channels == 1:
            model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = Identity()
        return model
    except Exception as e:
        logging.error(f"Failed to load pretrained ResNet model: {e}", exc_info=True)
        raise

def load_models(
    model_paths: Dict[str, str],
    device: torch.device,
    num_classes: int
) -> Tuple[nn.Module, nn.Module, nn.Module]:
    """
    Loads pretrained models from specified paths.
    """
    try:
        image_model_feat = load_pretrained_resnet_as_feature_extractor()
        channels_model_feat = load_pretrained_resnet_as_feature_extractor()
        sss_model_feat = load_pretrained_resnet_as_feature_extractor(input_channels=1)

        loaded_models = {
            "image": image_model_feat,
            "channels": channels_model_feat,
            "sss": sss_model_feat
        }

        for model_key, model in loaded_models.items():
            try:
                path = model_paths.get(model_key)
                if path and os.path.exists(path):
                    state_dict = torch.load(path, map_location=device)
                    model.load_state_dict(state_dict)
                    logging.info(f"{model_key.capitalize()} model loaded successfully from {path}.")
                else:
                    logging.warning(f"Path not found for model: {model_key} -> {path}")
            except Exception as inner_e:
                logging.error(f"Failed to load {model_key} model from {path}: {inner_e}", exc_info=True)

        return image_model_feat, channels_model_feat, sss_model_feat

    except Exception as e:
        logging.error(f"Error loading models: {e}", exc_info=True)
        raise
