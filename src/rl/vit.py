import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from utils.hyperparams import HPARAMS


class Encoder(nn.Module):
    '''
    Vision transformer encoder
    '''
    def __init__(self,) -> None:
        # Initialize the parameters
        self.pe = PositionalEncodings()
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=HPARAMS['rl']['vit']['patch_size'] ** 2,
                nhead=HPARAMS['rl']['vit']['nhead'],
                dim_feedforward=HPARAMS['rl']['vit']['dim_feedforward'],
                dropout=HPARAMS['rl']['dropout'],
                activation=F.gelu,
            ),
            num_layers=6,
        )
        
        
    def forward(
        self,
        pixels: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Generates a context vector based on the pixels
        This vector is used to train the decoder, and later, the policy
        
        Args:
            pixels (Tensor): Unflattened pixel tensor of the image
            
        Returns:
            out (Tensor): Context vector
        '''
        # Patch images across width and height
        # Then flatten the last dimensions
        patches = pixels.unfold(
            dimension=-2,
            size=HPARAMS['rl']['vit']['patch_size'],
            step=HPARAMS['rl']['vit']['patch_size'],
        ).unfold(
            dimension=-1,
            size=HPARAMS['rl']['vit']['patch_size'],
            step=HPARAMS['rl']['vit']['patch_size'],
        ).flatten(-2, -1)
        # Apply PEs
        patches = self.pe(patches)
        # Feed through encoder
        return self.encoder(patches)
        

class PositionalEncodings(nn.Module):
    '''
    Learned positional encodings for vision transformer
    '''
    def __init__(self,) -> None:
        # Initialize distribution used to sample params
        distribution = dist.Uniform(
            -1 / HPARAMS['rl']['vit']['patch_size'],
            1 / HPARAMS['rl']['vit']['patch_size'],
        )
        # Trainable parameters
        self.encodings = nn.Parameter(
            distribution.sample((
                HPARAMS['rl']['vit']['patch_size'] ** 2,
                HPARAMS['rl']['vit']['image_size'][0] * HPARAMS['rl']['vit']['image_size'][0],
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
            out (Tensor): Positionally encoded patches
        '''
        return x + self.encodings