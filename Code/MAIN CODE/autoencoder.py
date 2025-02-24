import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


# Alpha Scaling Layer
class AlphaScalingLayer(nn.Module):
    """
    A PyTorch module that performs alpha scaling on abundance values.
    Args:
        None
    Attributes:
        None
    Methods:
        forward(abundances, alpha): Performs alpha scaling on the input abundance values.
    """
    
    def forward(self, abundances, alpha):
        """
        Performs alpha scaling on the input abundance values.
        Args:
            abundances (torch.Tensor): The input abundance values.
            alpha (float): The scaling factor.
        Returns:
            torch.Tensor: The scaled abundance values.
        """
        
        alpha = torch.as_tensor(alpha, dtype=abundances.dtype, device=abundances.device)
        alpha = alpha.view(-1, 1, 1, 1)
        scaled_abundances = abundances * alpha
        
        return scaled_abundances

# Sum to One Layer
class SumToOne(nn.Module):
    """
    A PyTorch module that applies softmax function to the input tensor along the second dimension,
    scaling the input by a given scale factor.
    Args:
        scale (float): The scale factor to multiply the input tensor by before applying softmax.
    Inputs:
        - x (torch.Tensor): The input tensor of shape (batch_size, num_classes).
    Returns:
        - output (torch.Tensor): The output tensor after applying softmax to the scaled input tensor.
          It has the same shape as the input tensor.
    """
    
    def __init__(self, scale):
        super(SumToOne, self).__init__()
        self.scale = scale

    def forward(self, x):
        x = torch.nn.functional.softmax(self.scale * x, dim=1)
        return x

# Encoder
class Encoder(nn.Module):    
    def __init__(self, params):
        """
        This class represents an encoder module for an autoencoder.
        Args:
            params (dict): A dictionary containing the parameters for the encoder.
        Attributes:
            hidden_layer_one (nn.Conv2d): The first hidden layer of the encoder, which performs convolution.
            hidden_layer_two (nn.Conv2d): The second hidden layer of the encoder, which performs convolution.
            bn1 (nn.BatchNorm2d): Batch normalization layer for the first hidden layer.
            bn2 (nn.BatchNorm2d): Batch normalization layer for the second hidden layer.
            asc_layer (SumToOne): A custom layer that performs sum-to-one normalization.
        """

        super(Encoder, self).__init__()
        self.hidden_layer_one = nn.Conv2d(in_channels=params['n_bands'],
                                          out_channels=params['e_filters'],
                                          kernel_size=params['e_size'],
                                          stride=1, padding='same')
        self.hidden_layer_two = nn.Conv2d(in_channels=params['e_filters'],
                                          out_channels=params['num_endmembers'],
                                          kernel_size=1, stride=1, padding='same')
        self.bn1 = nn.BatchNorm2d(params['e_filters'])  # Initialize BatchNorm in constructor
        self.bn2 = nn.BatchNorm2d(params['num_endmembers'])
        self.asc_layer = SumToOne(params['scale'])
        
        nn.init.normal_(self.hidden_layer_one.weight, mean=0.0, std=0.03)
        nn.init.normal_(self.hidden_layer_two.weight, mean=0.0, std=0.03)

    def forward(self, x):
        """
        Forward pass of the encoder.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The output tensor after passing through the encoder.
        """
        
        x = self.hidden_layer_one(x)
        x = nn.LeakyReLU(0.02)(x)
        x = self.bn1(x)  # Use initialized BatchNorm
        x = nn.Dropout2d(0.2)(x)
        x = self.hidden_layer_two(x)
        x = nn.LeakyReLU(0.02)(x)
        x = self.bn2(x)  # Use initialized BatchNorm
        x = nn.Dropout2d(0.2)(x)
        x = self.asc_layer(x)
        return x

class NonNegativeConv2d(nn.Module):    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        """
        A custom convolutional layer that enforces non-negativity constraints on its weights.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            stride (int, optional): Stride of the convolution. Default is 1.
            padding (int, optional): Padding added to the input. Default is 0.
            bias (bool, optional): If True, adds a learnable bias to the output. Default is True.
        Attributes:
            weight_raw (nn.Parameter): The raw weights of the convolutional layer.
            bias (nn.Parameter or None): The learnable bias of the convolutional layer, if bias is True.
            stride (int): The stride of the convolution.
            padding (int): The padding added to the input.
        Methods:
            forward(x): Performs a forward pass of the convolutional layer.
        Note:
            This layer applies a rectified linear unit (ReLU) activation function to ensure that the weights are non-negative.
        """
        
        super(NonNegativeConv2d, self).__init__()
        # Initialize raw weights without constraints
        self.weight_raw = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.03)
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels) * 0.03)
        else:
            self.register_parameter('bias', None)
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # Apply ReLU to ensure weights are non-negative
        weight = F.relu(self.weight_raw)
        return F.conv2d(x, weight, self.bias, stride=self.stride, padding=self.padding)

# Decoder
class Decoder(nn.Module):
    def __init__(self, params):
        """
        Decoder module for an autoencoder.
        Args:
            params (dict): A dictionary containing the parameters for the decoder.
        Attributes:
            output_layer (NonNegativeConv2d): The output layer of the decoder.
        """
        
        super(Decoder, self).__init__()
        self.output_layer = NonNegativeConv2d(
            in_channels=params['num_endmembers'],
            out_channels=params['d_filters'],
            kernel_size=params['d_size'],
            stride=1,
            padding='same'  # Ensure compatibility with NonNegativeConv2d
        )

    def forward(self, x):
        x = self.output_layer(x)
        recon = nn.ReLU()(x)
        return recon
    
class ResidualProjectionHead(nn.Module):
    def __init__(self, num_endmembers):
        """
        Autoencoder class for performing dimensionality reduction using skip connections.
        Args:
            num_endmembers (int): Number of endmembers in the input data.
        Attributes:
            fc1 (nn.Linear): First fully connected layer.
            fc2 (nn.Linear): Second fully connected layer.
            shortcut (nn.Linear): Skip connection layer.
        """
        
        super().__init__()
        self.fc1 = nn.Linear(num_endmembers, 128)
        self.fc2 = nn.Linear(128, 64)
        self.shortcut = nn.Linear(num_endmembers, 64)  # Skip connection

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x))) + self.shortcut(x)  # Residual connection

# Autoencoder
class Autoencoder(nn.Module):
    """
    Autoencoder neural network model.
    Args:
        params (dict): A dictionary containing the parameters for the model.
    Attributes:
        encoder (Encoder): The encoder module of the autoencoder.
        decoder (Decoder): The decoder module of the autoencoder.
        alpha_scaling_layer (AlphaScalingLayer): The alpha scaling layer of the autoencoder.
        projection_head (nn.Sequential): The projection head of the autoencoder.
    """
    
    def __init__(self, params):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.alpha_scaling_layer = AlphaScalingLayer()

        # Low-dimensional Projection Head - num_endmembers -> 64 ####
        # self.projection_head = nn.Sequential(
        #     nn.Linear(params['num_endmembers'], 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64)  # Keep latent space fixed at 64
        # )
        
        ### Skip Connection Projection Head - num_endmembers -> 64 ####
        # self.projection_head = ResidualProjectionHead(params['num_endmembers'])
        
        # # Bottleneck Projection Head - num_endmembers -> 64 ####
        # self.projection_head = nn.Sequential(
        #     nn.Linear(params['num_endmembers'], 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 64),  # Compressed bottleneck
        #     nn.ReLU(),
        #     nn.Linear(64, 128)   # Expand again
        # )
        
        ## Deeper Projection Head - num_endmembers -> 64 ####
        # self.projection_head = nn.Sequential(
        #     nn.Linear(params['num_endmembers'], 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64)  # Keep final output at 64
        # )
        
        # Original Projection Head - num_endmembers -> 64 ####
        self.projection_head = nn.Sequential(
            nn.Linear(params['num_endmembers'], 128),
            nn.ReLU(),
            nn.Linear(128, 64) # Increase latent space
        )
        
        # Initialize weights correctly
        self._initialize_weights()
        
    def _initialize_weights(self):
        """
        Initialize the weights of the model's modules.
        This function initializes the weights of the convolutional and linear layers
        using specific initialization methods (He initialization for Conv layers and
        Xavier initialization for linear layers).
        """
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')  # He initialization for Conv layers
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)  # Xavier initialization for linear layers
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x, alpha=None):
        """
        Forward pass of the autoencoder.
        Args:
            x (torch.Tensor): The input tensor.
            alpha (torch.Tensor, optional): The alpha tensor for alpha scaling. Defaults to None.
        Returns:
            torch.Tensor: The reconstructed output tensor.
            torch.Tensor: The projected embedding tensor.
        """
        abundances = self.encoder(x)
        if alpha is not None:
            scaled_abundances = self.alpha_scaling_layer(abundances, alpha)
        else:
            scaled_abundances = abundances
        
        reconstructed = self.decoder(scaled_abundances)
        
        # Convert to a single vector per sample: [N, num_endmembers]
        embedding = F.adaptive_avg_pool2d(abundances, (1, 1))  # [N, num_endmembers, 1, 1]
        embedding = torch.flatten(embedding, start_dim=1)  # [N, num_endmembers]
        projected_embedding = self.projection_head(embedding)
        
        return reconstructed, projected_embedding # embedding