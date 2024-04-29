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
from module import function
from torchvision.models import vit_h_14, ViT_H_14_Weights


# データの前処理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


for seed in range(10):
    # サポートデータセットのインスタンス化
    support = function.SupportDataset(root_dir='../data/raw/cifar100/data/', transform=transform, num_classes=5, support_per_class=1, random_seed=seed)

    # クエリデータセットのインスタンス化
    query = function.QueryDataset(root_dir='../data/raw/cifar100/data/', transform=transform, num_classes=5, support_per_class=1,query_per_class=30, random_seed=seed)

    # Test validation ratio
    ratio = 0.5
    val_size = int(ratio * len(query))
    query_size = len(query) - val_size
    # Split the dataset into training and validation sets
    torch.manual_seed(seed)
    query, val = torch.utils.data.random_split(query, [query_size, val_size])


    # データローダーの作成
    support_loader = DataLoader(support, batch_size=1, shuffle=False)
    val_loader = DataLoader(val, batch_size=1, shuffle=False)
    query_loader = DataLoader(query, batch_size=1, shuffle=False)




    # Get the weights of the pretrained model
    weights = ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1

    # Get the pretrained model
    model = vit_h_14(weights=weights)


    model.heads = nn.Sequential(nn.Linear(1280, 1000), 
                                nn.ReLU(),
                                nn.Linear(1000, 512),
                                nn.ReLU(),
                                nn.Linear(512, 5))

    # Make the parameter of the last layer trainable
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last layer
    for param in model.heads.parameters():
        param.requires_grad = True


    # Define the loss function and optimizer
    device = torch.device("cuda:1")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), weight_decay=1e-5, lr=0.00005)
    model = model.to(device)




    epochs = 100


    history_train = {'loss': [], 'accuracy': []}
    history_val = {'loss': [], 'accuracy': []}
    history_test = {'loss': [], 'accuracy': []}

    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        for inputs, labels in support_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total += labels.size(0)
            predicted = torch.max(outputs.data, 1)[1]
            correct += (predicted == labels.squeeze()).sum().item()
        
        history_train['loss'].append(loss.item())
        history_train['accuracy'].append(correct/total)
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
        
                val_loss += criterion(outputs, labels)
                total += labels.size(0)
                predicted = torch.max(outputs.data, 1)[1]
    
                correct += (predicted == labels.squeeze()).sum().item()
            val_loss = val_loss/len(val_loader)
            history_val['loss'].append(val_loss.item())
            history_val['accuracy'].append(correct/total)
            
        print(f'Epoch {epoch+1}/{epochs}, Loss: {val_loss:.4f}, Accuracy: {correct/total:.4f}')
        
        model.eval()
        query_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in query_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                query_loss += criterion(outputs, labels)
                total += labels.size(0)
                predicted = torch.max(outputs.data, 1)[1]

                correct += (predicted == labels.squeeze()).sum().item()
            query_loss = query_loss/len(query_loader)
            history_test['loss'].append(query_loss.item())
            history_test['accuracy'].append(correct/total)

        print(f'Loss: {query_loss:.4f}, Accuracy: {correct/total:.4f}')

    history = {'train': history_train, 'val': history_val, 'test': history_test}
    np.save(f'../data/result/fsl/history_{seed}.npy', history)