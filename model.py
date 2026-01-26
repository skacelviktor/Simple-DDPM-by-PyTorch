"""
U-Net Model for DDPM (Denoising Diffusion Probabilistic Models)

This code implements very simpified denoising network architecture used in DDPM.
"""

import torch.nn as nn
import torch
import numpy as np

class TimeEmbedding(nn.Module):
    """
    Time embedding

    The time embedding layer transforms a known time value into an n-dimensional time embedding.
    It uses sinusoidal embeddings as the initial representation.
    The embedding is then passed through two fully connected layers,
    with a SiLU activation function.
    """
    
    def __init__(self, dim, device):
        super(TimeEmbedding, self).__init__()
        #Dimensionality of the time embedding
        self.dim = dim
        
        # Two linear layers with SiLU activation
        self.lin1 = nn.Linear(dim, dim * 4)
        self.Silu = nn.SiLU()                
        self.lin2 = nn.Linear(dim * 4, dim)
        
        # Formula: inv_freq[i] = 10000^(-2i/dim) for i in [0, dim/2)
        # This creates exponentially spaced frequencies for positional encoding
        self.inv_freq = 10000 ** (torch.arange(start=0, end=self.dim // 2, dtype=torch.float32, device=device) / (self.dim // 2))

    def forward(self, time_steps):   
        # Repeat time steps across half of the embedding dimension and
        # scale them using inverse frequencies for sinusoidal encoding
        t_emb = time_steps[:, None].repeat(1, self.dim // 2) / self.inv_freq
        
        # Apply sinusoidal functions: sin(angles) and cos(angles)
        # Result shape: (batch_size, dim)
        t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)

        # Pass through layers for learned transformation of embeddings
        t_emb = self.lin1(t_emb)
        t_emb = self.Silu(t_emb)
        t_emb = self.lin2(t_emb)

        return t_emb
    
class ResBlock(nn.Module):
    """
    Residual block with time conditioning for DDPM.

    This block applies two convolutional layers with a residual (skip) connection.
    Unlike a standard residual block, it is conditioned on the diffusion timestep.

    Architecture:
    - Conv2d (3*3) → SiLU activation
    - Add time embedding:
        - SiLU activation → Linear layer
        - Added to the output of the first Conv2d
    - SiLU activation → Conv2d (3*3)
    - If input and output channels differ, apply Conv2d (1*1)
    - Add residual connection

    Time conditioning allows the model to adapt its behavior across different
    stages of the diffusion process, from early noisy stages to later clean stages.
    """
    
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super(ResBlock, self).__init__()
        
        # First convolutional block:
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1),
                                   nn.SiLU())
        
        # MLP for time embedding:
        self.time_mlp = nn.Sequential(nn.SiLU(),
                                      nn.Linear(time_emb_dim, out_ch))
        
        # Second convolutional block:
        self.conv2 = nn.Sequential(nn.SiLU(),
                                   nn.Conv2d(out_ch, out_ch, 3, padding=1))
        
        # Shortcut connection: if input and output channels differ, use 1x1 conv
        # Otherwise use identity (no change) since channels already match
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        # First convolution
        h = self.conv1(x)
        
        # Process time embedding and broadcast to spatial dimensions
        # Shape: (batch_size, out_ch) -> (batch_size, out_ch, 1, 1)
        # Broadcasting allows adding the same time info to all spatial locations
        time_emb = self.time_mlp(t)[:, :, None, None]
        
        # Add time conditioning to feature maps
        h = h + time_emb
        
        # Second convolution
        h = self.conv2(h)
        
        # Add residual connection, and pass shortcut (Conv2d)
        return h + self.shortcut(x)
    
class Down_block(nn.Module):
    """
    Downsampling block for the U-Net encoder.
    
    This block reduces dimensions by a factor of two while processing features
    through a ResBlock. Skip connections are saved for later use in the Up_block (decoder).
    
    Architecture:
    - ResBlock
    - MaxPool2d (2*2): downsampling (H, W) -> (H/2, W/2)
    """
    
    def __init__(self, in_channles, out_channels, t_dim):
        super(Down_block, self).__init__()
        
        # ResBlock
        self.Res_block = ResBlock(in_channles, out_channels, t_dim)
        
        # Max pooling for spatial downsampling
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t):
        # Apply ResBlock
        skip = self.Res_block(x, t)

        # Downsample for next encoder level
        x = self.pool(skip)

        # Return both the downsampled features and the skip connection for Up Block
        return x, skip
    
class Up_block(nn.Module):
    """
    Upsampling block for the U-Net decoder.
    
    This block increases dimensions by a factor of 2 and concatenates
    skip connections from the corresponding encoder level.
    
    Architecture:
    - ConvTranspose2d (2*2): upsampling (H, W) -> (2H, 2W)
    - Concatenation with skip connection
    - ResBlock
    """
    
    def __init__(self, in_channles, out_channels, t_dim):
        super(Up_block, self).__init__()

        # Transposed convolution for spatial upsampling
        # Stride 2 doubles the spatial dimensions
        self.up = nn.ConvTranspose2d(in_channles, in_channles, kernel_size=2, stride=2)

        # ResBlock that processes concatenated features (skip + upsampled)
        # Input channels = in_channels * 2 (skip + upsampled concatenated)
        self.Res_block = ResBlock(in_channles*2, out_channels, t_dim)

    def forward(self, x, skip, t):
        # Upsample features
        x = self.up(x)
        
        # Concatenate skip connection along channel dimension
        # This combines fine-grained details from encoder with coarse features from decoder
        x = torch.cat([x, skip], dim=1)
        
        # Process through ResBlock
        x = self.Res_block(x, t)

        return x

class U_net(nn.Module):
    """
    U-Net Architecture for DDPM Noise Prediction.
    
    A symmetric encoder-decoder network with skip connections. 
    The model predicts noise given a noisy image and timestep.
    
    Architecture:
    - Input: Noisy image (1, 28, 28) + timestep
    - Encoder: 2 downsampling blocks reducing spatial dims: 28→14→7
    - Bottleneck: ResBlock at lowest resolution (7x7)
    - Decoder: 2 upsampling blocks restoring spatial dims: 7→14→28
    - Output: Predicted noise (1, 28, 28)
    
    Skip Connections: Features from each encoder level are concatenated with
    corresponding decoder levels.
    """
    
    def __init__(self, device, in_channels = 1, t_dim = 256):
        super(U_net, self).__init__()

        # Time embedding module: converts timestep to vector
        self.time_mlp = TimeEmbedding(t_dim, device)

        # Encoder: downsamples while increasing channel count
        # Level 0: (1, 28, 28) -> (64, 14, 14)
        # Level 1: (64, 14, 14) -> (128, 7, 7)
        self.Down_blocks = nn.ModuleList([
            Down_block(in_channels, 64, t_dim),   # 28x28 -> 14x14, 1ch -> 64ch
            Down_block(64, 128, t_dim)             # 14x14 -> 7x7, 64ch -> 128ch
        ])
        
        # Bottleneck:
        self.bottleneck = ResBlock(128, 128, t_dim)

        # Decoder: upsamples while decreasing channel count
        # Uses skip connections from encoder levels
        # Level 0: (128, 7, 7) + skip(128, 14, 14) -> (64, 14, 14)
        # Level 1: (64, 14, 14) + skip(64, 28, 28) -> (64, 28, 28)
        self.Up_blocks = nn.ModuleList([
            Up_block(128, 64, t_dim),   # 7x7 -> 14x14, 256ch -> 64ch
            Up_block(64, 64, t_dim)     # 14x14 -> 28x28, 128ch -> 64ch
        ])
        
        # Final output layer:
        self.final_conv = nn.Conv2d(64, in_channels, kernel_size=1)

    def forward(self, x, t):      
        # Generate time embeddings
        t_emb = self.time_mlp(t)

        # Encoder: downsampling path
        # Store skip connections from each level for later use in decoder
        skips = []
        for down_block in self.Down_blocks:
            x, skip = down_block(x, t_emb)
            skips.append(skip)

        # Bottleneck:
        x = self.bottleneck(x, t_emb)

        # Decoder: upsampling path with skip connections
        # Process skip connections in reverse order
        for idx, skip in enumerate(reversed(skips)):
            x = self.Up_blocks[idx](x, skip, t_emb)

        # Final projection:
        x = self.final_conv(x)

        return x