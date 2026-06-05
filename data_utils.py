"""
Data loading utilities for CIFAR-100.
Handles loading, normalization, and coreset selection methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os


def get_cifar100_data(data_dir='./data'):
    """Load CIFAR-100 and return normalized tensors.
    
    Returns:
        train_images: [50000, 3, 32, 32] float tensor, normalized
        train_labels: [50000] long tensor
        test_loader: DataLoader for test set
        channel_mean: per-channel mean
        channel_std: per-channel std
    """
    from datasets import load_dataset
    
    cache_path = os.path.join(data_dir, 'cifar100_tensors.pt')
    if os.path.exists(cache_path):
        print("Loading cached CIFAR-100 tensors...")
        data = torch.load(cache_path, weights_only=True)
        train_images = data['train_images']
        train_labels = data['train_labels']
        test_images = data['test_images']
        test_labels = data['test_labels']
        channel_mean = data['channel_mean']
        channel_std = data['channel_std']
    else:
        print("Loading CIFAR-100 from HuggingFace...")
        ds_train = load_dataset('uoft-cs/cifar100', split='train', trust_remote_code=True)
        ds_test = load_dataset('uoft-cs/cifar100', split='test', trust_remote_code=True)
        
        # Convert to tensors
        train_images = torch.tensor(np.array([np.array(img) for img in ds_train['img']]), 
                                     dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        train_labels = torch.tensor(ds_train['fine_label'], dtype=torch.long)
        
        test_images = torch.tensor(np.array([np.array(img) for img in ds_test['img']]), 
                                    dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        test_labels = torch.tensor(ds_test['fine_label'], dtype=torch.long)
        
        # Compute channel-wise mean and std
        channel_mean = train_images.mean(dim=[0, 2, 3], keepdim=True)
        channel_std = train_images.std(dim=[0, 2, 3], keepdim=True)
        
        # Normalize
        train_images = (train_images - channel_mean) / channel_std
        test_images = (test_images - channel_mean) / channel_std
        
        channel_mean = channel_mean.squeeze()
        channel_std = channel_std.squeeze()
        
        # Cache
        os.makedirs(data_dir, exist_ok=True)
        torch.save({
            'train_images': train_images,
            'train_labels': train_labels,
            'test_images': test_images,
            'test_labels': test_labels,
            'channel_mean': channel_mean,
            'channel_std': channel_std,
        }, cache_path)
        print(f"Cached tensors to {cache_path}")
    
    # Create test loader
    test_dataset = TensorDataset(test_images, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    
    return train_images, train_labels, test_loader, channel_mean, channel_std


def get_class_indices(labels, num_classes=100):
    """Get indices for each class."""
    class_indices = {}
    for c in range(num_classes):
        class_indices[c] = (labels == c).nonzero(as_tuple=True)[0]
    return class_indices


def random_selection(train_images, train_labels, ipc, num_classes=100, seed=42):
    """Random coreset selection: randomly select IPC images per class."""
    rng = np.random.RandomState(seed)
    class_indices = get_class_indices(train_labels, num_classes)
    
    selected_indices = []
    for c in range(num_classes):
        indices = class_indices[c].numpy()
        chosen = rng.choice(indices, size=ipc, replace=False)
        selected_indices.extend(chosen.tolist())
    
    selected_indices = torch.tensor(selected_indices)
    return train_images[selected_indices], train_labels[selected_indices]


def k_centers_selection(train_images, train_labels, ipc, num_classes=100, device='cpu'):
    """K-centers coreset selection using pixel-space farthest-point sampling."""
    class_indices = get_class_indices(train_labels, num_classes)
    
    selected_indices = []
    for c in range(num_classes):
        indices = class_indices[c]
        class_images = train_images[indices].reshape(len(indices), -1)  # Flatten
        
        # Greedy farthest-point sampling
        chosen = []
        # Start with the image closest to the class mean
        mean = class_images.mean(dim=0)
        dists = torch.cdist(class_images.unsqueeze(0), mean.unsqueeze(0).unsqueeze(0)).squeeze()
        first_idx = dists.argmin().item()
        chosen.append(first_idx)
        
        # Min distance to chosen set
        min_dists = torch.cdist(class_images.unsqueeze(0), 
                                class_images[first_idx].unsqueeze(0).unsqueeze(0)).squeeze()
        
        for _ in range(1, ipc):
            next_idx = min_dists.argmax().item()
            chosen.append(next_idx)
            
            new_dists = torch.cdist(class_images.unsqueeze(0),
                                    class_images[next_idx].unsqueeze(0).unsqueeze(0)).squeeze()
            min_dists = torch.minimum(min_dists, new_dists)
        
        selected_indices.extend(indices[chosen].tolist())
    
    selected_indices = torch.tensor(selected_indices)
    return train_images[selected_indices], train_labels[selected_indices]


def generate_soft_labels(train_images, train_labels, test_loader, 
                         num_classes=100, device='cuda', 
                         n_teachers=1, teacher_epochs=200):
    """Train teacher model(s) and generate soft labels for the training set.
    
    Returns:
        soft_labels: [N, num_classes] logits from teacher ensemble
    """
    from networks import get_convnet
    from augmentations import DiffAugment
    from evaluate import evaluate_model
    
    all_logits = torch.zeros(train_images.shape[0], num_classes, device=device)
    
    for t in range(n_teachers):
        print(f"  Training teacher {t+1}/{n_teachers}...")
        model = get_convnet('cifar100', num_classes=num_classes).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=teacher_epochs)
        criterion = nn.CrossEntropyLoss()
        
        images_device = train_images.to(device)
        labels_device = train_labels.to(device)
        n = images_device.shape[0]
        batch_size = 256
        
        for epoch in range(teacher_epochs):
            model.train()
            indices = torch.randperm(n)
            
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_idx = indices[start:end]
                imgs = images_device[batch_idx]
                lbls = labels_device[batch_idx]
                
                imgs = DiffAugment(imgs, strategy='color_crop_cutout_flip_scale_rotate')
                
                logits = model(imgs)
                loss = criterion(logits, lbls)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            
            if (epoch + 1) % 50 == 0:
                acc = evaluate_model(model, test_loader, device)
                print(f"    Teacher {t+1}, Epoch {epoch+1}: {acc:.2f}%")
                model.train()
        
        # Generate logits
        model.eval()
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                imgs = images_device[start:end]
                logits = model(imgs)
                all_logits[start:end] += logits
    
    all_logits /= n_teachers
    return all_logits.cpu()


if __name__ == '__main__':
    print("Loading CIFAR-100...")
    train_images, train_labels, test_loader, mean, std = get_cifar100_data()
    print(f"Train: {train_images.shape}, Labels: {train_labels.shape}")
    print(f"Mean: {mean}, Std: {std}")
    
    # Test random selection
    sel_imgs, sel_lbls = random_selection(train_images, train_labels, ipc=10)
    print(f"Random selection IPC=10: {sel_imgs.shape}, {sel_lbls.shape}")
    
    # Test k-centers selection
    sel_imgs_kc, sel_lbls_kc = k_centers_selection(train_images, train_labels, ipc=10)
    print(f"K-centers selection IPC=10: {sel_imgs_kc.shape}, {sel_lbls_kc.shape}")
    
    print("Data utilities work!")
