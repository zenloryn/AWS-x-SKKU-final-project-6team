import os
from torchvision import datasets
from torch.utils.data import DataLoader
from train.transforms import get_train_transforms, get_val_transforms
import torch

def collate(batch):
    # batch: list of (image_tensor, label_int_or_tensor)
    images, labels = zip(*batch)  # tuples
    images = torch.stack(images, dim=0)  # [B, C, H, W]

    # labels가 int 튜플이면 tensor로, tensor면 그대로 stack
    if not torch.is_tensor(labels[0]):
        labels = torch.tensor(labels, dtype=torch.float32)  # [B]
    else:
        labels = torch.stack(labels, dim=0).to(torch.float32)  # [B] 또는 [B,1] → [B]가 되면 좋음

    # 여기서 [B]로 맞춰 둡니다. (학습 루프에서 BCEWithLogitsLoss와 [N] 로짓에 맞춰 사용)
    if labels.ndim > 1:
        labels = labels.view(-1)

    return images, labels

def get_dataloaders(img_size, batch_size):
    train_dir = os.environ.get("SM_CHANNEL_TRAIN", "./data/train")
    val_dir   = os.environ.get("SM_CHANNEL_VAL", "./data/val")

    # Transform (Train/Val 분리)
    train_tf = get_train_transforms(img_size)
    val_tf   = get_val_transforms(img_size)

    # Dataset
    train_dataset = datasets.ImageFolder(train_dir, transform=train_tf)
    val_dataset   = datasets.ImageFolder(val_dir,   transform=val_tf)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True,  collate_fn=collate)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate)

    return train_loader, val_loader, train_dataset


def get_dataloaders_test(img_size, batch_size):
    # 채널 경로(없으면 로컬 기본 경로로 폴백)
    test_dir  = os.environ.get("SM_CHANNEL_TEST", "./data/test")

    # Transform
    val_tf_test   = get_val_transforms(img_size)

    test_dataset  = datasets.ImageFolder(test_dir,  transform=val_tf_test)

    # DataLoader
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate)

    return test_loader