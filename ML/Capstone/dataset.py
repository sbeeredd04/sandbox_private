import torch
from torch.utils.data import Dataset

class AttributeFilterDataset(Dataset):
    """Filter dataset to only include selected attributes"""
    
    def __init__(self, base_dataset, attribute_indices):
        self.base_dataset = base_dataset
        self.attribute_indices = attribute_indices

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, attrs = self.base_dataset[idx]
        filtered_attrs = attrs[self.attribute_indices]
        return img, filtered_attrs
