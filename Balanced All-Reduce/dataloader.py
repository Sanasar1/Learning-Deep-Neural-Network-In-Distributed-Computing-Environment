import torch
from torchvision import transforms
from torch.utils.data import random_split, DataLoader, Subset
import torchvision
import torch.distributed as dist
import numpy as np
import time

def get_loaders(batch_size, world_size, rank, model, device):
    full_trainset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=transforms.ToTensor())

    means = (full_trainset.data.astype(np.float32) / 255).mean(axis=(0, 1, 2))
    stds = (full_trainset.data.astype(np.float32) / 255).std(axis=(0, 1, 2))
    train_transforms = transforms.Compose(
    [
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds)
    ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=means, std=stds)
        ]
    )

    full_trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=train_transforms)
    testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=test_transforms)

    # Splitting the full training set into training and validation sets
    train_size = int(0.8 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    trainset, valset = random_split(full_trainset, [train_size, val_size])

    # Estimate the epoch duration for adaptive data partitioning
    trainloader_temp = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    local_efficiency_ratio, efficiency_ratios = estimate_epoch_duration(trainloader_temp, world_size, model, device, num_batches=10)

    dataset_partrition_train, train_indices = get_dataset_partition(trainset, rank, efficiency_ratios)
    trainloader = torch.utils.data.DataLoader(dataset_partrition_train, batch_size=batch_size, shuffle=False)

    # Dataset and DataLoader for validation
    dataset_partrition_val, val_indices = get_dataset_partition(valset, rank, efficiency_ratios)
    val_loader = torch.utils.data.DataLoader(dataset_partrition_val, batch_size=batch_size, shuffle=False)

    # DataLoader for testing
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, val_loader, test_loader, trainset, valset, train_indices, val_indices

def get_dataset_partition(dataset, rank, efficiency_ratios):
    # Total Dataset Size
    total_size = len(dataset)

    print(total_size)

    start_idx = 0
    i = 0
    
    for ratio in efficiency_ratios:
        # Calculate the number of data points for this node
        num_data_points = int(total_size * ratio)
        
        # Define the start and end indices for this node
        end_idx = start_idx + num_data_points
        
        if i==rank:
            indices = list(range(start_idx, end_idx))
            print(f'Rank {rank} indices: {(start_idx, end_idx)}')
            return Subset(dataset, range(start_idx, end_idx)), indices
        
        start_idx = end_idx
        i += 1

def get_next_dataset_partition(dataset, prev_indices, prev_fraction, next_fraction, epoch_duration, epoch_duration_tensor):
    total_size = len(dataset)

    # Calculate the efficiency ratio
    efficiency_ratio = epoch_duration / epoch_duration_tensor

    # Calculate the total number of data points this node should handle based on its efficiency
    node_data_points = int(total_size * efficiency_ratio)

    # Calculate the number of data points from the previous subset
    prev_subset_size = int(node_data_points * prev_fraction)

    # Calculate the number of data points for the next subset
    next_subset_size = int(node_data_points * next_fraction)

    # Select indices for the previous subset
    prev_subset_indices = np.random.choice(prev_indices, size=prev_subset_size, replace=False)

    # Determine remaining indices after selecting for the previous subset
    remaining_indices = list(set(range(total_size)) - set(prev_subset_indices))

    # Select indices for the next subset
    next_subset_indices = np.random.choice(remaining_indices, size=next_subset_size, replace=False)

    # Combine the two subsets to get the final subset for this epoch
    final_indices = list(prev_subset_indices) + list(next_subset_indices)

    return Subset(dataset, final_indices), final_indices


def get_subset_loaders(trainset, valset, train_indices, val_indices, batch_size, prev_fraction, next_fraction, epoch_duration, epoch_duration_tensor):
    # Get the new subset of the dataset
    subset_train, indices_train = get_next_dataset_partition(trainset, train_indices, prev_fraction, next_fraction, epoch_duration, epoch_duration_tensor)
    subset_val, indices_val = get_next_dataset_partition(valset, val_indices, prev_fraction, next_fraction, epoch_duration, epoch_duration_tensor)

    # Create a DataLoader for this subset
    loader_train = torch.utils.data.DataLoader(subset_train, batch_size=batch_size, shuffle=False)
    loader_val = torch.utils.data.DataLoader(subset_val, batch_size=batch_size, shuffle=False)

    # Return the DataLoader and the indices used, so they can be used in the next epoch
    return loader_train, loader_val, indices_train, indices_val

def estimate_epoch_duration(trainloader, world_size, model, device, num_batches=10):
    start_time = time.time()
    model.to(device)
    model.train()

    # Iterate over a limited number of batches to simulate training
    for i, (inputs, targets) in enumerate(trainloader):
        if i >= num_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = model(inputs)
        
        outputs.sum().backward()
        
    end_time = time.time()
    
    local_duration = end_time - start_time

    # Prepare for gathering durations across all nodes
    local_duration_tensor = torch.tensor([local_duration], device=device)
    all_durations_tensor = [torch.tensor([0.0], device=device) for _ in range(world_size)]
    
    # Gather durations from all nodes
    dist.all_gather(all_durations_tensor, local_duration_tensor)
    
    # Convert tensor list to a list of floats and calculate total duration
    all_durations = [tensor.item() for tensor in all_durations_tensor]
    total_duration = sum(all_durations)

    # Calculate efficiency ratios
    local_efficiency_ratio = local_duration / total_duration
    efficiency_ratios = [duration / total_duration for duration in all_durations]

    return local_efficiency_ratio, efficiency_ratios