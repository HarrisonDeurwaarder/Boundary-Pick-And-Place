import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from utils.config import load_config


CONFIG = load_config('panda_train')


class Encoder(nn.Module):
    '''
    Vision transformer encoder
    Generates a depth map 
    '''
    def __init__(self,) -> None:
        super.__init__()
        # Initialize the parameters
        self.pe = PositionalEncodings()
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=CONFIG['rl']['vit']['patch_size'] ** 2,
                nhead=CONFIG['rl']['vit']['nhead'],
                dim_feedforward=CONFIG['rl']['vit']['dim_feedforward_encoder'],
                dropout=CONFIG['rl']['dropout'],
                activation=F.gelu,
            ),
            num_layers=6,
        )
        self.dense = nn.Sequential(
            nn.Linear(
                CONFIG['rl']['vit']['image_size'][0] * CONFIG['rl']['vit']['image_size'][1],
                CONFIG['rl']['vit']['dim_feedforward_conv'],
            ),
            nn.ReLU(),
            nn.Linear(
                CONFIG['rl']['vit']['dim_feedforward_conv'],
                CONFIG['rl']['vit']['patch_size']
            )
        )
        # Construct the depth map and logvars (variance = inverse certainty)
        self.conv = nn.ConvTranspose2d(
                in_channels=1,
                out_channels=2, # Depth and logvar maps
                kernel_size=CONFIG['rl']['vit']['kernel_size'],
            )
        
        
    def forward(
        self,
        pixels: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Generates a depth map
        
        Args:
            pixels (Tensor): Unflattened pixel tensor of the image
            
        Returns:
            tuple[Tensor]: A tuple containing:
                - depths: Predicted depth map of the RGB image
                - logvars: Corresponding log-variances of the predicted depth distributions
        '''
        # Patch images across width and height
        # Then flatten the last dimensions
        patches: torch.Tensor = pixels.unfold(
            dimension=-2,
            size=CONFIG['rl']['vit']['patch_size'],
            step=CONFIG['rl']['vit']['patch_size'],
        ).unfold(
            dimension=-1,
            size=CONFIG['rl']['vit']['patch_size'],
            step=CONFIG['rl']['vit']['patch_size'],
        ).flatten(-2, -1)
        # Apply PEs
        patches = self.pe(patches)
        # Feed through encoder
        out: torch.Tensor = self.encoder(patches)[..., 0, :] # Extract fixed-size latent representation
        # Final dense layer and transposed convolution
        out = self.conv(
            self.dense(out),
        )
        # Split depths and logvars
        return out[..., 0], out[..., 1]
    
    
    @classmethod
    def vit_objective(
        cls,
        depths: torch.Tensor,
        target_depths: torch.Tensor,
        logvars: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Evaluates the loss of the depth and variance (inverse confidence) maps using Gaussian NLL loss
        
        Args:
            depths (Tensor): Predicted depth map based on the RGB scene image
            target_depths (Tensor): Target depth map using sensor data
            logvars (Tensor): Predicted (log) variance of estimation (inverse of confidence)
            
        Returns:
            loss: Gaussian negative log-likelihood lose optimizing both predictions and logvars
        '''
        return F.gaussian_nll_loss(
            depths,
            target_depths,
            torch.exp(logvars),
        )
        

class PositionalEncodings(nn.Module):
    '''
    Learned positional encodings for vision transformer
    '''
    def __init__(self,) -> None:
        super().__init__()
        # Initialize distribution used to sample params
        distribution = dist.Uniform(
            -1 / CONFIG['rl']['vit']['patch_size'],
            1 / CONFIG['rl']['vit']['patch_size'],
        )
        # Trainable parameters
        self.encodings = nn.Parameter(
            distribution.sample((
                CONFIG['rl']['vit']['patch_size'] ** 2,
                CONFIG['rl']['vit']['image_size'][0] * CONFIG['rl']['vit']['image_size'][0],
            ))
        )
        
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Apply positional encodings to patches
        
        Args:
            x (Tensor): Flattened patches
            
        Returns:
            Tensor: Positionally encoded patches
        '''
        return x + self.encodings