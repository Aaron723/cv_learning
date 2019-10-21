# -*- coding: utf-8 -*-
# @Time    : 2019/10/18 18:07
# @Author  : Ziqi Wang
# @FileName: con_as_feature_extractor.py
# @Email: zw280@scarletmail.rutgers.edu
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import DataLoader
from torchvision import transforms, datasets, models

from transfer_learning import train_model, binary_cross_entropy_with_logits

"""
in this practice, the resnet act like a feature extractor, only the last fc will be trained
so it is like resnet18 + fc(customized)
"""

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
    model_conv = models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 12)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_conv = model_conv.to(device)

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = torch.optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
    criterion = binary_cross_entropy_with_logits
    model_conv = train_model(model_conv, dataloader, criterion, optimizer_conv,
                             exp_lr_scheduler, num_epochs=25, device=device)
