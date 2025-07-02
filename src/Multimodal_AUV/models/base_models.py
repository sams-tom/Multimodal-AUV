import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F
import torch


class ResNet50Custom(nn.Module):
        def __init__(self, input_channels, num_classes):
            super(ResNet50Custom, self).__init__()
        
            # Store input_channels as an attribute for later access
            self.input_channels = input_channels
        
            # Load pretrained ResNet50 model
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)  # Equivalent to pretrained=True
        
            # Modify the first convolutional layer to accept custom input channels
            self.model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
            # Replace the final fully connected layer to match the number of classes
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        def forward(self, x):
            # Forward pass through the model
            return self.model(x)

        def get_feature_size(self):
            # Return the size of the feature map (before the final fully connected layer)
            return self.model.fc.in_features

class Identity(nn.Module):
    def forward(self, x):
        return x

class AdditiveAttention(nn.Module):
        def __init__(self, d_model, hidden_dim=128):
            super(AdditiveAttention, self).__init__()
            self.query_projection = nn.Linear(d_model, hidden_dim)
            self.key_projection = nn.Linear(d_model, hidden_dim)
            self.value_projection = nn.Linear(d_model, hidden_dim)
            self.attention_mechanism = nn.Linear(hidden_dim, hidden_dim) # Output hidden_dim

        def forward(self, query):
            keys = self.key_projection(query)
            values = self.value_projection(query)
            queries = self.query_projection(query)

            attention_scores = torch.tanh(queries + keys)
            attention_weights = F.softmax(self.attention_mechanism(attention_scores), dim=1) # Softmax across hidden dim

            attended_values = values * attention_weights # No sum here!
            return attended_values

class MultiModalModel(nn.Module):
        def __init__(self, image_model_feat, bathy_model_feat, sss_model_feat, num_classes, attention_type="scaled_dot_product"):  # Add attention_type
            super(MultiModalModel, self).__init__()
            self.image_model_feat = image_model_feat
            self.bathy_model_feat = bathy_model_feat
            self.sss_model_feat = sss_model_feat
            self.fc = nn.Linear(384, 1284)
            self.fc1 = nn.Linear(1284, 32)
            num_classes = int(num_classes)
            if not isinstance(num_classes, int):
                raise TypeError("num_classes must be an integer")  # Raise a clear error
            self.fc2 = nn.Linear(32, num_classes)
            self.attention_type = attention_type  # Store the attention type

  
            self.attention_image = AdditiveAttention(2048)
            self.attention_bathy = AdditiveAttention(2048)
            self.attention_sss = AdditiveAttention(2048)
            # Add more attention types as needed

        def forward(self, inputs, bathy_tensor, sss_image):

            image_features = self.image_model_feat(inputs)

            bathy_features = self.bathy_model_feat(bathy_tensor)

            sss_features = self.sss_model_feat(sss_image)

            image_features_attended = self.attention_image(image_features)
            bathy_features_attended = self.attention_bathy(bathy_features)
            sss_features_attended = self.attention_sss(sss_features)

            combined_features = torch.cat([image_features_attended, bathy_features_attended, sss_features_attended], dim=1)
            outputs_1 = self.fc(combined_features)
            output_2 = self.fc1(outputs_1)
            outputs = self.fc2(output_2)
            return outputs