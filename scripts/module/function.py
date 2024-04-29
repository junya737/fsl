import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim

class SupportDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_classes=5, support_per_class=1, class_seed=42, transform_seed=42):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_seed = class_seed
        self.transform_seed = transform_seed
        
        # fix seed for class selection
        random.seed(self.class_seed)
        all_classes = sorted(os.listdir(root_dir))
        self.selected_classes = random.sample(all_classes, num_classes)

        for idx, cls in enumerate(self.selected_classes):
            cls_folder = os.path.join(root_dir, cls)
            img_names = os.listdir(cls_folder)
            selected_imgs = random.sample(img_names, support_per_class)

            for img_name in selected_imgs:
                self.images.append(os.path.join(cls_folder, img_name))
                self.labels.append(idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        # fix seed for transform
        random.seed(self.transform_seed)
        if self.transform:
            image = self.transform(image)
        # reset seed
        random.seed(self.class_seed)
        
        label = self.labels[idx]
        return image, label

class QueryDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_classes=5, support_per_class=1, query_per_class=15, class_seed=42):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        random.seed(class_seed)
        all_classes = sorted(os.listdir(root_dir))
        self.selected_classes = random.sample(all_classes, num_classes)  # 選択されたクラス

        for idx, cls in enumerate(self.selected_classes):  # idxは0-4の範囲
            cls_folder = os.path.join(root_dir, cls)
            img_names = os.listdir(cls_folder)

            # サポート画像を除外
            support_imgs = random.sample(img_names, support_per_class)
            remaining_imgs = [img for img in img_names if img not in support_imgs]

            # 残りの画像から指定された数のクエリ画像を選択
            if len(remaining_imgs) > query_per_class:
                query_imgs = random.sample(remaining_imgs, query_per_class)
            else:
                query_imgs = remaining_imgs  # 残りの画像が指定数より少ない場合はすべて選択

            for img_name in query_imgs:
                self.images.append(os.path.join(cls_folder, img_name))
                self.labels.append(idx)  # idxは0-4の範囲

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


class ImageClassificationDataset(Dataset):
    def __init__(self, root_dir, class_labels, transform=None):
        """
        Args:
            root_dir (string): データセットが存在するディレクトリのパス。
            class_labels (dict): クラス名をキー、ラベルを値とする辞書。
        """
        self.root_dir = root_dir
        self.class_labels = class_labels
        self.transform = transform
        self.data = []
        self.load_dataset()

    def load_dataset(self):
        """ディレクトリから画像とラベルを読み込む"""
        for class_name, label in self.class_labels.items():
            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.png'):
                    img_path = os.path.join(class_dir, img_name)
                    self.data.append((img_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label