# -*- coding: utf-8 -*-
# @Time    : 2019/10/10 17:06
# @Author  : Ziqi Wang
# @FileName: DataLoader.py
# @Email: zw280@scarletmail.rutgers.edu
import os
from skimage import io
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.models import resnet18
from random import randint
from skimage import io, transform
import numpy as np


class img_dataset(Dataset):
    """
    image dataset
    """

    def __init__(self, root_dir, transform=None):
        """
        :param root_dir: the root path of both imgs and ground truth
        :param transform: transform of the imag and tag
        """
        self.root_dir = root_dir
        self.transform = transform
        # self.img_transform = transforms

    def __len__(self):
        path = os.path.abspath(os.path.dirname(__file__))
        count = len(os.listdir(path + "\data"))
        return int(count / 2)

    def __getitem__(self, idx=0):
        """
        :param idx: the index of the video
        :return: the dict of path of video and path of tag
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        video_name = self.root_dir + "\\" + str(idx)
        tag_name = self.root_dir + "\\" + str(idx) + ".txt"
        # with open(tag_name, 'r') as file:
        #     lines = file.readlines()
        sample = {'video_name': video_name, 'tag_name': tag_name}
        pair = self.get_imgs(sample=sample)
        return pair

    def read_tag(self, file):
        lines = list()
        with open(file, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if line[-1] == "\n":
                    line = line[:-1]
                line = list(map(int, line.split(",")))
                line = np.array(line)
                lines.append(line)
            f.close()
        return lines

    def get_imgs(self, sample):
        video_name = sample['video_name']
        tag_name = sample['tag_name']
        # sample_list = list()
        # with open(tag_name, 'r') as file:
        #     lines = file.readlines()
        #     file.close()
        lines = self.read_tag(tag_name)
        start_point = randint(0, len(lines) - 1)
        # end_point = start_point + 64
        # for i in range(start_point, end_point):
        #    img_name = video_name + "\{}.png".format(i)
        #    tag = lines[i]
        #    img = io.imread(img_name)
        #    pair = {"img": img, "tag": tag}
        #    if self.transform:
        #        pair = self.transform(pair)
        #    sample_list.append(pair)
        img_name = video_name + "\{}.png".format(start_point)
        tag = lines[start_point]
        img = io.imread(img_name)
        pair = {"img": img, "tag": tag}
        if self.transform:
            pair = self.transform(pair)
        pair['image'] = pair['image'][0]
        return pair


class Rescale(object):
    """
    Rescale the image in a sample to a given size
    """

    def __init__(self, output_size):
        """

        :param output_size: Desired output size. If it is tuple, output is matched to output_size.
        If int, smaller edge of image is matched to output_size keeping aspect ratio the same
        """
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, tag = sample["img"], sample["tag"]
        #   h stand for height and w stand for width
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        return {"img": img, "tag": tag}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, tag = sample['img'], sample['tag']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        # randomly add some value on edge, new edge is between given number and original ones
        image = image[top: top + new_h,
                left: left + new_w]
        return {"img": image, "tag": tag}


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors
    """

    def __call__(self, sample):
        image, tag = sample['img'], sample['tag']
        image = image.transpose((2, 0, 1))
        image = ((torch.from_numpy(image)).unsqueeze(0)).float()
        tag = ((torch.from_numpy(tag)).unsqueeze(0)).float()
        return {'image': image,
                'tag': tag}


class Normalize(object):
    """
    normalize the given data
    """
    def __call__(self, sample, mean, std):
        pass

if __name__ == '__main__':
    dataset = img_dataset(root_dir="F:\img_training\data", transform=transforms.Compose([
        Rescale(256),
        RandomCrop(224),
        ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    i = 0
    # batch size == 4, so the length of dataloader = length of dataset / 4 = 9
    for item in dataloader:
        # .to(device)
        img = item['image'].to(device)
        print(i)
        tag = item['tag'].to(device)
        i += 1
        print(img.size(), tag.size())
    # dataiter = iter(dataloader)
    # img = next(dataiter)["image"]
    # tag = next(dataiter)["tag"]
    #    imgs, labels = next(dataiter)

