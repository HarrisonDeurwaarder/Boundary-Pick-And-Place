import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from datasets import load_dataset


class NYUV2Dataset(Dataset):
    '''
    Dataset for the NYU-v2 rgb-to-depth dataset
    '''
    def __init__(self,) -> None:
        data = load_dataset('jagennath-hari/nyuv2')
        self.train_transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((100, 100),),
            v2.ToDtype(torch.float32, scale=True,),
            v2.GaussianNoise(sigma=0.005,),
        ])
        self.test_transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((100, 100),),
            v2.ToDtype(torch.float32, scale=True,),
            v2.GaussianNoise(sigma=0.02,),
        ])
        # Extract training set
        self.train = data['train'].with_transform(self.apply_train_transforms,)
        self.test = data['test'].with_transform(self.apply_test_transforms,)
        
        
    def __len__(self,) -> int:
        return len(self.train)
    
    
    def __getitem__(
        self, 
        idx: int,
    ) -> tuple:
        return (
            self.train['rgb'][idx],
            self.train['depth'][idx],
        )
        
        
    def apply_train_transforms(self, batch,) -> torch.Tensor:
        # Apply transforms
        batch['rgb'], batch['depth'] = self.train_transform(batch['rgb'], batch['depth'])
        return batch
        
        
    def apply_test_transforms(self, batch,) -> torch.Tensor:
        # Apply transforms
        batch['rgb'], batch['depth'] = self.test_transform(batch['rgb'], batch['depth'])
        return batch