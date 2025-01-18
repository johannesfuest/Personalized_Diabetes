import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    """
    PyTorch equivalent of the custom ConvLayer from TensorFlow code.
    This layer applies two conv->bn->pool->drop sequences and then flattens.
    """

    def __init__(self, CONV_INPUT_LENGTH: int, fixed_hyperparameters):
        super(ConvLayer, self).__init__()
        self.CONV_INPUT_LENGTH = CONV_INPUT_LENGTH

        # Extract hyperparameters
        f1 = fixed_hyperparameters["filter_1"]
        k1 = fixed_hyperparameters["kernel_1"]
        s1 = fixed_hyperparameters["stride_1"]
        p1_pool_size = fixed_hyperparameters["pool_size_1"]
        p1_pool_stride = fixed_hyperparameters["pool_stride_1"]

        f2 = fixed_hyperparameters["filter_2"]
        k2 = fixed_hyperparameters["kernel_2"]
        s2 = fixed_hyperparameters["stride_2"]
        p2_pool_size = fixed_hyperparameters["pool_size_2"]
        p2_pool_stride = fixed_hyperparameters["pool_stride_2"]

        dropout_rate = fixed_hyperparameters["dropout_rate"]

        # First Conv sequence
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=f1, kernel_size=k1, stride=s1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm1d(num_features=f1)
        self.pool1 = nn.MaxPool1d(kernel_size=p1_pool_size, stride=p1_pool_stride, padding=0)
        self.drop1 = nn.Dropout(p=dropout_rate)

        # Second Conv sequence
        self.conv2 = nn.Conv1d(in_channels=f1, out_channels=f2, kernel_size=k2, stride=s2, padding=0, bias=False)
        self.bn2 = nn.BatchNorm1d(num_features=f2)
        self.pool2 = nn.MaxPool1d(kernel_size=p2_pool_size, stride=p2_pool_stride, padding=0)
        self.drop2 = nn.Dropout(p=dropout_rate)

        # Flatten layer
        self.flatten = nn.Flatten()

    def forward(self, x):
        # x is of shape (batch, 1, CONV_INPUT_LENGTH)
        assert x.shape[2] == self.CONV_INPUT_LENGTH, "Input length does not match CONV_INPUT_LENGTH."

        # First conv block
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Second conv block
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # Flatten the output
        x = self.flatten(x)
        return x


class GlucoseModel(nn.Module):
    """
    PyTorch model replicating the architecture defined in the TensorFlow code.
    """

    def __init__(self, CONV_INPUT_LENGTH: int, self_sup: bool, fixed_hyperparameters):
        super(GlucoseModel, self).__init__()
        self.CONV_INPUT_LENGTH = CONV_INPUT_LENGTH
        self.self_sup = self_sup
        self.fixed_hyperparameters = fixed_hyperparameters

        # Four ConvLayers for the four inputs
        self.diabetes_conv = ConvLayer(CONV_INPUT_LENGTH, fixed_hyperparameters)
        self.meal_conv = ConvLayer(CONV_INPUT_LENGTH, fixed_hyperparameters)
        self.smbg_conv = ConvLayer(CONV_INPUT_LENGTH, fixed_hyperparameters)
        self.exercise_conv = ConvLayer(CONV_INPUT_LENGTH, fixed_hyperparameters)

        dropout_rate = fixed_hyperparameters["dropout_rate"]

        self.fc1 = nn.Linear(168, 512)
        self.drop_fc1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.drop_fc2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(256, 128)
        self.drop_fc3 = nn.Dropout(p=dropout_rate)
        self.fc4 = nn.Linear(128, 64)

        self.output_layer = nn.Linear(64, 4)
        self.fc5 = nn.Linear(64, 1)

    def forward(self, x):
        # x expected shape: (batch, 1, CONV_INPUT_LENGTH*4)
        # Slice into four equal parts along the length dimension
        bs = x.shape[0]
        # Each segment is CONV_INPUT_LENGTH long
        seg_len = self.CONV_INPUT_LENGTH

        diabetes_input = x[:, :, 0:seg_len]
        meal_input = x[:, :, seg_len:2*seg_len]
        smbg_input = x[:, :, 2*seg_len:3*seg_len]
        exercise_input = x[:, :, 3*seg_len:4*seg_len]
        
        # Pass through the four ConvLayers
        diabetes_conv_out = self.diabetes_conv(diabetes_input)
        meal_conv_out = self.meal_conv(meal_input)
        smbg_conv_out = self.smbg_conv(smbg_input)
        exercise_conv_out = self.exercise_conv(exercise_input)

        # Concatenate
        post_conv = torch.cat([diabetes_conv_out, meal_conv_out, smbg_conv_out, exercise_conv_out], dim=1)

        # Fully connected layers
        x = self.fc1(post_conv)
        x = F.relu(x)
        x = self.drop_fc1(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.drop_fc2(x)

        x = self.fc3(x)
        x = F.relu(x)
        x = self.drop_fc3(x)

        x = self.fc4(x)
        x = F.relu(x)

        if self.self_sup:
            x = self.output_layer(x)
        else:
            x = self.fc5(x)
        return x
