import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

from datasets import load_dataset
from datetime import datetime

from source.utils.config import load_config

load_config('train')

from source.core.sl.dataset import NYUV2Dataset
from source.core.sl.vit import Encoder
import source.utils.config as config

# dense.1
def main() -> None:
    '''
    Main function
    '''
    # Set deterministic seed
    seed = torch.seed()
    dataset: NYUV2Dataset = NYUV2Dataset()
    # Batch the data
    dataloader: DataLoader = DataLoader(
        dataset=dataset,
        batch_size=config.config['sl']['batch_size'],
        shuffle=True,
    )
    
    model: Encoder = Encoder()
    op: optim.Adam = optim.Adam(
        params=model.parameters(),
        lr=config.config['sl']['lr'],
    )
    
    # Training loop
    for epoch in range(config.config['sl']['epochs']):
        for rgb, depth in dataloader:
            op.zero_grad()
            # Compute loss
            out, logvar = model(rgb)
            loss: torch.Tensor = Encoder.vit_objective(out, depth, logvar)
            # Backpropagate
            loss.backward()
            op.step()
        # Get test tensors
        test_rgb: torch.Tensor = torch.stack([spec['rgb'] for spec in dataset.test])
        test_depth: torch.Tensor = torch.stack([spec['depth'] for spec in dataset.test])
        with torch.inference_mode():
            # Metrics
            depth, logvar = model(test_rgb)
            mae: torch.Tensor = torch.abs(torch.abs(depth - test_depth)).mean()
            mse: torch.Tensor = torch.square(torch.abs(depth - test_depth)).mean()
            nll: torch.Tensor = Encoder.vit_objective(depth, test_depth, logvar)
            print(
                '='*30,
                f'Epoch {epoch} completed.',
                f'Test MAE: {mae}',
                f'Test MSE: {mse}',
                f'Test NLL + {config.config["sl"]["mse_coef"]}MSE: {nll}',
                sep='\n'
            )
    # Save model
    timestamp: str = datetime.now().strftime('%Y%m%d')
    torch.save(
        model.state_dict(),
        f'source\\models\\depthnet\\depth-inference-{timestamp}-loss{nll:.2f}-seed{seed}.pt'
    )
    
    
if __name__ == '__main__':
    main()