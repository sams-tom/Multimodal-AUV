import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin # Import the mixin

# --- Custom Model Definitions ---

class Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

class AdditiveAttention(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = 128):
        super(AdditiveAttention, self).__init__()
        self.query_projection = nn.Linear(d_model, hidden_dim)
        self.key_projection = nn.Linear(d_model, hidden_dim)
        self.value_projection = nn.Linear(d_model, hidden_dim)
        self.attention_mechanism = nn.Linear(hidden_dim, hidden_dim) # Output hidden_dim

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        keys = self.key_projection(query)
        values = self.value_projection(query)
        queries = self.query_projection(query)

        attention_scores = torch.tanh(queries + keys)
        attention_weights = F.softmax(self.attention_mechanism(attention_scores), dim=1)

        attended_values = values * attention_weights # Element-wise product
        return attended_values

class ResNet50Custom(nn.Module, PyTorchModelHubMixin): # Inherit from PyTorchModelHubMixin
    def __init__(self, input_channels: int, num_classes: int, **kwargs):
        super(ResNet50Custom, self).__init__()

        # Store config for PyTorchModelHubMixin to serialize to config.json
        self.config = {
            "input_channels": input_channels,
            "num_classes": num_classes,
            **kwargs
        }

        self.input_channels = input_channels

        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        self.model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # The final FC layer of ResNet50Custom will be used *only* when ResNet50Custom is a standalone classifier.
        # When used as a feature extractor within MultiModalModel, this layer will be temporarily replaced by Identity().
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_feature_size(self) -> int:
        return self.model.fc.in_features


class MultiModalModel(nn.Module, PyTorchModelHubMixin): # Inherit from PyTorchModelHubMixin
    def __init__(self,
                 image_input_channels: int,
                 bathy_input_channels: int,
                 sss_input_channels: int,
                 num_classes: int,
                 attention_type: str = "scaled_dot_product",
                 **kwargs): # Added **kwargs for mixin compatibility
        super(MultiModalModel, self).__init__()

        # Store config for PyTorchModelHubMixin to serialize to config.json
        self.config = {
            "image_input_channels": image_input_channels,
            "bathy_input_channels": bathy_input_channels,
            "sss_input_channels": sss_input_channels,
            "num_classes": num_classes,
            "attention_type": attention_type,
            **kwargs # Pass along any extra kwargs for mixin
        }

        # Instantiate feature extraction models *inside* MultiModalModel
        # Their final FC layers will be treated as Identity for feature extraction
        self.image_model_feat = ResNet50Custom(input_channels=image_input_channels, num_classes=num_classes)
        self.bathy_model_feat = ResNet50Custom(input_channels=bathy_input_channels, num_classes=num_classes)
        self.sss_model_feat = ResNet50Custom(input_channels=sss_input_channels, num_classes=num_classes)

        # The ResNet50's feature output size is 2048 before its final FC layer
        feature_dim = self.image_model_feat.get_feature_size() # Should be 2048

        # Attention layers (AdditiveAttention uses d_model and outputs hidden_dim)
        attention_hidden_dim = 128 # This matches your fc layer input calculation (3*128=384)
        self.attention_image = AdditiveAttention(feature_dim, hidden_dim=attention_hidden_dim)
        self.attention_bathy = AdditiveAttention(feature_dim, hidden_dim=attention_hidden_dim)
        self.attention_sss = AdditiveAttention(feature_dim, hidden_dim=attention_hidden_dim)

        # Final classification layers
        self.fc = nn.Linear(3 * attention_hidden_dim, 1284)
        self.fc1 = nn.Linear(1284, 32)
        # Ensure num_classes is int for the linear layer
        num_classes_int = int(num_classes)
        if not isinstance(num_classes_int, int):
            raise TypeError("num_classes must be an integer after casting")
        self.fc2 = nn.Linear(32, num_classes_int)
        self.attention_type = attention_type

    def forward(self, inputs: torch.Tensor, bathy_tensor: torch.Tensor, sss_image: torch.Tensor) -> torch.Tensor:
        # Temporarily replace the final FC layer of the feature extractors with Identity
        # to get the 2048 features, then restore them.
        original_image_fc = self.image_model_feat.model.fc
        original_bathy_fc = self.bathy_model_feat.model.fc
        original_sss_fc = self.sss_model_feat.model.fc

        self.image_model_feat.model.fc = Identity()
        self.bathy_model_feat.model.fc = Identity()
        self.sss_model_feat.model.fc = Identity()

        image_features = self.image_model_feat(inputs)
        bathy_features = self.bathy_model_feat(bathy_tensor)
        sss_features = self.sss_model_feat(sss_image)

        # Restore original FC layers on the feature extractors
        self.image_model_feat.model.fc = original_image_fc
        self.bathy_model_feat.model.fc = original_bathy_fc
        self.sss_model_feat.model.fc = original_sss_fc

        # Apply attention
        image_features_attended = self.attention_image(image_features)
        bathy_features_attended = self.attention_bathy(bathy_features)
        sss_features_attended = self.attention_sss(sss_features)

        # Concatenate attended features
        combined_features = torch.cat([image_features_attended, bathy_features_attended, sss_features_attended], dim=1)

        # Pass through final classification layers
        outputs_1 = self.fc(combined_features)
        output_2 = self.fc1(outputs_1)
        outputs = self.fc2(output_2)
        return outputs