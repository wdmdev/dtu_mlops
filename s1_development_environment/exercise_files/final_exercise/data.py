import os
import torch
from torch.utils.data import Dataset, DataLoader

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, image_tensors, target_tensors):
        self.images = image_tensors
        self.labels = target_tensors

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx].unsqueeze(0)
        y = self.labels[idx]
        return x, y

# Function to create a dataset from the given list of file names
def create_dataset(image_files, target_files, data_path):
    image_tensors = [torch.load(data_path + file_name) for file_name in image_files]
    target_tensors = [torch.load(data_path + file_name) for file_name in target_files]
    dataset = CustomDataset(torch.cat(image_tensors, dim=0), torch.cat(target_tensors, dim=0))
    return dataset


def mnist():
    """Return train and test dataloaders for MNIST."""
    data_path = os.path.join(os.path.dirname(__file__), 
                             "..", "..", "..", "data", "corruptmnist","")

    # Create file name lists
    train_image_files = [f'train_images_{i}.pt' for i in range(6)]
    train_target_files = [f'train_target_{i}.pt' for i in range(6)]
    test_image_files = ['test_images.pt']
    test_target_files = ['test_target.pt']

    # Create datasets
    train_dataset = create_dataset(train_image_files, train_target_files, data_path)
    test_dataset = create_dataset(test_image_files, test_target_files, data_path)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=None, shuffle=False)


    return train_loader, test_loader