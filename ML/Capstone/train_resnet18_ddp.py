"""
Distributed Data Parallel (DDP) Training Script for ResNet18 on CelebA
This script uses multiple GPUs to train the model, reducing memory usage per GPU
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
import numpy as np
import os
import argparse

# Import custom dataset
from dataset import AttributeFilterDataset


# Configuration
class Config:
    image_size = 224
    batch_size = 512  # This will be divided by number of GPUs (4 GPUs = 128 per GPU)
    num_workers = 16
    data_dir = './data'
    num_epochs = 20
    learning_rate = 0.001
    weight_decay = 1e-4
    selected_attributes = ['Heavy_Makeup', 'Wearing_Lipstick', 'Attractive', 'High_Cheekbones', 'Rosy_Cheeks']
    num_attributes = len(selected_attributes)
    gpu_ids = "1,2,3,4"  # Use GPUs 1, 2, 3, 4 only


# ResNet18 Model Architecture
class BasicBlock(nn.Module):
    """Basic residual block with skip connection"""
    
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        # First conv layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second conv layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out


class ResNet18MultiLabel(nn.Module):
    """ResNet18 architecture for multi-label classification"""
    
    def __init__(self, num_classes=5):
        super(ResNet18MultiLabel, self).__init__()
        
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers (2, 2, 2, 2 blocks)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


# Distributed Training Functions
def setup_distributed(rank, world_size):
    """Initialize distributed training environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group(
        backend='nccl',  # Use NCCL for GPU training
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, rank):
    """Train for one epoch with DDP support"""
    model.train()
    running_loss = 0.0
    all_predictions = []
    all_targets = []

    for batch_idx, (data, target) in enumerate(train_loader):
        # Move to device
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).float()

        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        running_loss += loss.item()
        with torch.no_grad():
            predictions = torch.sigmoid(output) > 0.5
            all_predictions.append(predictions.cpu())
            all_targets.append(target.cpu())

        # Progress (only print from rank 0)
        if rank == 0 and batch_idx % 10 == 0:
            print(f'\rEpoch {epoch}: [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}', end='', flush=True)

    # Calculate metrics
    epoch_loss = running_loss / len(train_loader)
    all_predictions = torch.cat(all_predictions).numpy()
    all_targets = torch.cat(all_targets).numpy()
    epoch_acc = (all_predictions == all_targets).mean()

    # Gather metrics from all processes
    loss_tensor = torch.tensor([epoch_loss], device=device)
    acc_tensor = torch.tensor([epoch_acc], device=device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(acc_tensor, op=dist.ReduceOp.SUM)
    epoch_loss = loss_tensor.item() / dist.get_world_size()
    epoch_acc = acc_tensor.item() / dist.get_world_size()

    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion, device):
    """Validate the model with DDP support"""
    model.eval()
    val_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True).float()

            output = model(data)
            loss = criterion(output, target)

            val_loss += loss.item()
            predictions = torch.sigmoid(output) > 0.5
            all_predictions.append(predictions.cpu())
            all_targets.append(target.cpu())

    # Calculate metrics
    val_loss /= len(val_loader)
    all_predictions = torch.cat(all_predictions).numpy()
    all_targets = torch.cat(all_targets).numpy()
    val_acc = (all_predictions == all_targets).mean()

    # Gather metrics from all processes
    loss_tensor = torch.tensor([val_loss], device=device)
    acc_tensor = torch.tensor([val_acc], device=device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(acc_tensor, op=dist.ReduceOp.SUM)
    val_loss = loss_tensor.item() / dist.get_world_size()
    val_acc = acc_tensor.item() / dist.get_world_size()

    return val_loss, val_acc


def prepare_datasets(config):
    """Prepare train and validation datasets"""
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(config.image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load datasets
    train_dataset = datasets.CelebA(
        root=config.data_dir,
        split='train',
        transform=train_transform,
        download=False,
        target_type='attr'
    )
    
    val_dataset = datasets.CelebA(
        root=config.data_dir,
        split='valid',
        transform=val_transform,
        download=False,
        target_type='attr'
    )

    # Get attribute indices
    attribute_names = [name for name in train_dataset.attr_names if name.strip()]
    attribute_indices = [attribute_names.index(attr) for attr in config.selected_attributes]

    # Wrap with filter
    train_dataset = AttributeFilterDataset(train_dataset, attribute_indices)
    val_dataset = AttributeFilterDataset(val_dataset, attribute_indices)

    return train_dataset, val_dataset


def main_worker(rank, world_size, config):
    """Main training function for each process"""
    # Setup distributed training
    setup_distributed(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"Starting Distributed Training on {world_size} GPUs")
        print(f"{'='*80}")
    
    # Prepare datasets
    train_dataset, val_dataset = prepare_datasets(config)
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Calculate per-GPU batch size
    per_gpu_batch_size = config.batch_size // world_size
    per_gpu_workers = max(1, config.num_workers // world_size)
    
    # Create dataloaders with distributed samplers
    train_loader = DataLoader(
        train_dataset,
        batch_size=per_gpu_batch_size,
        sampler=train_sampler,
        num_workers=per_gpu_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=per_gpu_batch_size,
        sampler=val_sampler,
        num_workers=per_gpu_workers,
        pin_memory=True
    )
    
    # Create model and move to device
    model = ResNet18MultiLabel(num_classes=config.num_attributes).to(device)
    
    # Wrap model with DDP
    model = DDP(
        model,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=False
    )
    
    # Loss, optimizer, and scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=2,
        verbose=(rank == 0),
        min_lr=1e-7
    )
    
    if rank == 0:
        print(f"\n✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"✓ Batch size per GPU: {per_gpu_batch_size}")
        print(f"✓ Effective total batch size: {config.batch_size}")
        print(f"✓ Training batches per epoch: {len(train_loader)}")
        print(f"✓ Validation batches per epoch: {len(val_loader)}")
        print(f"{'='*80}\n")
    
    # Training loop
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0.0
    
    for epoch in range(1, config.num_epochs + 1):
        # Set epoch for sampler (important for proper shuffling)
        train_sampler.set_epoch(epoch)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, rank
        )
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler (only on rank 0)
        if rank == 0:
            scheduler.step(val_acc)
            
            # Store metrics
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # Print epoch summary
            print(f'\n{"-"*80}')
            print(f'Epoch {epoch}/{config.num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
            print(f'  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
            print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
            print(f'{"-"*80}')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Save the underlying model (not the DDP wrapper)
                torch.save(model.module.state_dict(), 'best_celeba_resnet18_ddp.pth')
                print(f'  ✓ New best model saved with Val Acc: {best_val_acc:.4f}')
    
    if rank == 0:
        print(f"\n{'='*80}")
        print("Training Complete!")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        print(f"Model saved as: best_celeba_resnet18_ddp.pth")
        print(f"{'='*80}\n")
    
    # Cleanup
    cleanup_distributed()


def main():
    """Main entry point"""
    # Restrict to specific GPUs (1, 2, 3, 4)
    # This must be set before CUDA initialization
    gpu_ids = "1,2,3,4"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Distributed Training for ResNet18 on CelebA')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=512, help='Total batch size across all GPUs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gpu-ids', type=str, default=gpu_ids, help='Comma-separated GPU IDs to use (default: 1,2,3,4)')
    args = parser.parse_args()
    
    # Update CUDA_VISIBLE_DEVICES if provided via command line
    if args.gpu_ids != gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    
    # Create config
    config = Config()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    
    # Check available GPUs (after setting CUDA_VISIBLE_DEVICES)
    world_size = torch.cuda.device_count()
    if world_size < 1:
        print("Error: No GPUs detected!")
        print(f"CUDA_VISIBLE_DEVICES was set to: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        return
    
    print(f"Using GPU IDs: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print(f"Found {world_size} GPU(s) available for training")
    print(f"\nConfiguration:")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Total batch size: {config.batch_size}")
    print(f"  Batch size per GPU: {config.batch_size // world_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Attributes: {config.selected_attributes}")
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Spawn processes for distributed training
    mp.spawn(
        main_worker,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    main()
