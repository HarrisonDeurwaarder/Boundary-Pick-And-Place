import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

import source.utils.config as config


class Encoder(nn.Module):
    '''
    Vision transformer encoder
    Generates a depth map 
    '''
    def __init__(self,) -> None:
        super().__init__()
        # Initialize the parameters
        self.pe = PositionalEncodings()
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=3 * config.config['sl']['vit']['patch_size'] ** 2,
                nhead=config.config['sl']['vit']['encoder']['n_head'],
                dim_feedforward=config.config['sl']['vit']['encoder']['dim_feedforward_encoder'],
                dropout=config.config['sl']['vit']['dropout'],
                activation=F.gelu,
                batch_first=True,
            ),
            num_layers=6,
        )
        self.dense = nn.Sequential(
            nn.Linear(
                3 * (config.config['sl']['vit']['patch_size'] ** 2) * (config.config['sl']['vit']['image_size'][0] * config.config['sl']['vit']['image_size'][1]) // (config.config['sl']['vit']['patch_size'] ** 2),
                config.config['sl']['vit']['encoder']['dim_feedforward_conv'],
            ),
            nn.ReLU(),
            nn.Linear(
                config.config['sl']['vit']['encoder']['dim_feedforward_conv'],
                config.config['sl']['vit']['image_size'][0] * config.config['sl']['vit']['image_size'][1]
            )
        )
        # Construct the depth map and logvars (variance = inverse certainty)
        self.conv = nn.ConvTranspose2d(
                in_channels=1,
                out_channels=2, # Depth and logvar maps
                kernel_size=config.config['sl']['vit']['encoder']['kernel_size'],
                padding=1,
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
            size=config.config['sl']['vit']['patch_size'],
            step=config.config['sl']['vit']['patch_size'],
        ).unfold(
            dimension=-2,
            size=config.config['sl']['vit']['patch_size'],
            step=config.config['sl']['vit']['patch_size'],
        ).flatten(-2, -1).flatten(-3, -2).permute(0, 2, 3, 1).flatten(-2, -1)
        # Apply PEs
        patches = self.pe(patches)
        # Feed through encoder
        out: torch.Tensor = self.encoder(patches) # (B, 100, 300)
        out = out.flatten(-2, -1) # (B, 30000)
        # Dense layer
        out = self.dense(out).view(-1, 1, config.config['sl']['vit']['image_size'][0], config.config['sl']['vit']['image_size'][1])
        out = self.conv(out)
        # Split depths and logvars
        return out[:, 0, ...], out[:, 1, ...]
    
    
    @classmethod
    def vit_objective(
        cls,
        depths: torch.Tensor,
        target_depths: torch.Tensor,
        logvars: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Evaluates the loss of the depth and variance (inverse confidence) maps using Gaussian NLL loss with an MSE term to weigh correct predictions
        
        Args:
            depths (Tensor): Predicted depth map based on the RGB scene image
            target_depths (Tensor): Target depth
            logvars (Tensor): Predicted (log) variance of estimation (inverse of confidence)
            
        Returns:
            loss: Gaussian negative log-likelihood loss + MSE
        '''
        return F.gaussian_nll_loss(
            depths,
            target_depths.squeeze(-3),
            torch.exp(logvars),
        ) + config.config['sl']['mse_coef'] * F.mse_loss(
            depths,
            target_depths.squeeze(-3),
        )
        

class PositionalEncodings(nn.Module):
    '''
    Learned positional encodings for vision transformer
    '''
    def __init__(self,) -> None:
        super().__init__()
        # Initialize distribution used to sample params
        distribution = dist.Uniform(
            -1 / config.config['sl']['vit']['patch_size'],
            1 / config.config['sl']['vit']['patch_size'],
        )
        # Trainable parameters
        self.encodings = nn.Parameter(
            distribution.sample((
                config.config['sl']['vit']['image_size'][0] * config.config['sl']['vit']['image_size'][1] // (config.config['sl']['vit']['patch_size'] ** 2),
                3 * config.config['sl']['vit']['patch_size'] ** 2,
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