from __future__ import print_function
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
from tqdm import tqdm
from os.path import exists


def default_loader(path):
    return Image.open(path).convert("RGB")


def main():
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--model-path",
        type=str,
        default='',
        help="model path",
    )

    args = parser.parse_args()

    device = torch.device("cuda")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((64, 64)),

        ]
    )
    model_name_list = ['resnet101_batchsize_16','vgg19','efficientnetb7_batchsize_16']
    

    fh = open('test.txt', "r")
    imgs = []
    for line in fh:
        line = line.strip("\n")
        line = line.rstrip()
        words = line.split()

        label = int(words[2] == 'positive')

        # print(label)

        imgs.append(('test/' + words[1], label))

    acc = []
    for epoch in range(29,30):
        model_list = []
        for model_name in model_name_list:
            path = 'save_model/' + str(epoch) + '_'+ model_name +'_model.pt'
            model = torch.load(path)
            model_list.append(model)
        
        tp, tn, fp, fn = 0, 0, 0, 0
        for img, label in (imgs):
            sum_pred = torch.tensor([0., 0.], device = torch.device("cuda"))
            sum_pred.to(device)
            imgpath = img
            img = default_loader(img)
            img = transform(img)
            img = img[None, :]
            for model in model_list:
                model = model.to(device)
                model.eval()
                pred = model(img.to(device))
                # print(sum_pred.shape, pred.shape)
                sum_pred += pred[0][0:2]
                # print(pred[0][0:2])
            last_pred = torch.argmax(sum_pred)
            if last_pred == label:
                if last_pred == 0:
                    tn += 1
                else:
                    tp += 1
            else:
                
                if last_pred == 0:
                    print(imgpath, 'fn')
                    fn += 1
                else:
                    fp += 1
                    print(imgpath, 'fp')
        
        print(tp, tn, fp, fn)
        accuracy, precision, recall = 0, 0, 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        acc.append(accuracy)
        if (tp + fp) > 0:
            precision = tp / (tp + fp)
        if (tp + fn) > 0:
            recall = tp / (tp + fn)

        print('accuarcy = ', accuracy)
        print('precision = ', precision)
        print('recall = ', recall)


        

    # --- ---------------------------------------------------------------------------------------- ---
    # Train Result
    epoch = 30
    from datetime import datetime
    import matplotlib.pyplot as plt

    now = datetime.now().strftime("%m%d-%H%M%S")
    print(epoch)
    ep = range(epoch)
    plt.plot(ep, acc, 'b', label='Test accuracy')
    plt.title('test set accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(loc='lower right')

    plt.savefig('accuracy-' + now + '.png')
    plt.figure()


if __name__ == '__main__':
    main()