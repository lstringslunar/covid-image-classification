from __future__ import print_function

from datetime import datetime, time

import numpy as np
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau
import antialiased_cnns

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import sys


def default_loader(path):
    return Image.open(path).convert("RGB")


class COVID_Dataset(Dataset):
    def __init__(
            self,
            txt,
            transform=None,
            target_transform=None,
            loader=default_loader,
            image_folder="train/",
    ):
        fh = open(txt, "r")
        imgs = []

        for line in fh:
            line = line.strip("\n")
            line = line.rstrip()
            words = line.split()

            label = int(words[2] == 'positive')

            if image_folder == "train/":
                imgs.append((image_folder + words[1], label))
            else:
                imgs.append((image_folder + words[0], 0))

        print(len(imgs))
        from random import sample
        random.seed(12345)
        sample_imgs = sample(imgs, 1000)
        print(len(sample_imgs))

        self.imgs = sample_imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


def train(
        args,
        model,
        device,
        train_loader,
        optimizer,
        epoch,

):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(F.log_softmax(output, dim=1), target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if args.dry_run:
                break


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        metavar="N",
        help="input batch size for training (default: 100)",
    )

    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.5,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="disables CUDA training"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        metavar="S",
        help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    # model = antialiased_cnns.resnext101_32x8d(pretrained=True)
    # model = models.alexnet(pretrained=True)
    model = models.resnet50(pretrained=True)
    # print(model)
    # model = models.alexnet(pretrained=True)
    # model.fc = nn.Linear(2048, 2)
    model.fc = nn.Linear(2048, 2)
    model = model.to(device)

    # augmentation
    transform_aug = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=(0.485, 0.456, 0.406),
            #     std=(0.229, 0.224, 0.225)
            #     ),
            # transforms.RandomRotation(20, resample=Image.BICUBIC, expand=True)
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((64, 64)),
            # transforms.RandomAffine(
            #     degrees=(-20, 20),
            #     translate=(0.15, 0.15),
            #     scale=(0.9, 1.1),
            #     shear=(0.2)
            # ),
            # transforms.CenterCrop(300),
        ]
    )

    train_data = COVID_Dataset(txt="train.txt", transform=transform_aug)
    train_loader = DataLoader(
        train_data,
        args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=1e-5, max_lr=1e-3, cycle_momentum=False
    )
    torch.save(model, 'save_model/' + str(0) + '_model.pt')
    # train
    for epoch in range(1, args.epochs + 1):
        train(
            args,
            model,
            device,
            train_loader,
            optimizer,
            epoch
        )
        scheduler.step()

        torch.save(model, 'save_model/' + str(epoch) + '_model.pt')


if __name__ == "__main__":
    main()
