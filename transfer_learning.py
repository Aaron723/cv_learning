# -*- coding: utf-8 -*-
# @Time    : 2019/10/13 3:05
# @Author  : Ziqi Wang
# @FileName: transfer_learning.py
# @Email: zw280@scarletmail.rutgers.edu


"""
in this practice, we initialize the resnet18 with a pretrained model, and train it by ourselves(all the layers will be trained)
another method is to freeze the layers except the last one so the pretrained model will act like a feature extractor
"""

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import DataLoader
from torchvision import transforms, datasets, models
import time
import copy
from torch.nn.functional import binary_cross_entropy_with_logits

data_dir = 'F:\img_training\data'
data_transform = {
    'train': transforms.Compose([
        DataLoader.Rescale(256),
        DataLoader.RandomCrop(224),
        # transforms.Resize(224, 224),
        DataLoader.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # transforms.ToPILImage
    ])}
image_dataset = DataLoader.img_dataset(root_dir=data_dir, transform=data_transform['train'])
dataloader = DataLoader.DataLoader(image_dataset, batch_size=4, shuffle=True, num_workers=4)

dataset_sizes = len(image_dataset)


def train_model(model, dataloader, criterion, optimizer, scheduler, num_epochs=25, device=None):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epoch = 0
    while True:
        # for epoch in range(num_epochs):
        print('Epoch{}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        phase = 'train'
        model.train()  # set model to training mode
        running_loss = 0.0
        running_corrects = 0

        for item in dataloader:
            inputs = item['image'].to(device)
            labels = torch.squeeze(item['tag'].to(device))
            # inputs = item['image'].to(device)
            # labels = item['tag'].to(device)

            optimizer.zero_grad()
            running_loss = 0
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                # labels[:, 0, ...]
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                _, y = torch.max(labels, 1)
                running_corrects += torch.sum(preds == y)
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / dataset_sizes
            epoch_acc = running_corrects.double() / dataset_sizes

            # scheduler.step()
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                'train', epoch_loss, epoch_acc))
  #      if epoch_loss <= 0.2:
 #           break
        print()
        epoch += 1
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    data_dir = 'F:\img_training\data'
    data_transform = {
        'train': transforms.Compose([
            DataLoader.Rescale(256),
            DataLoader.RandomCrop(224),
            # transforms.Resize(224, 224),
            DataLoader.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # transforms.ToPILImage
        ])}
    image_dataset = DataLoader.img_dataset(root_dir=data_dir, transform=data_transform['train'])
    dataloader = DataLoader.DataLoader(image_dataset, batch_size=4, shuffle=True, num_workers=4)

    dataset_sizes = len(image_dataset)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # detect the input dimenson of full connected nn of model, then we can define the output dimension of the last fc,(according to the ground truth)

    model_ft.fc = nn.Linear(num_ftrs, 12)
    model_ft = model_ft.to(device)
    criterion = binary_cross_entropy_with_logits
    # decay lr by a factor of 0.1 every 7 epochs
    optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.00001, momentum=0.9)
    # decay lr by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, dataloader, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=25, device=device)
