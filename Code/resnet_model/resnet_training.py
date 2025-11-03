import numpy as np 
import pandas as pd
import math
from PIL import Image
from tqdm import tqdm
import torchvision
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR, CosineAnnealingWarmRestarts
import copy
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

from torchvision import transforms
from monai.networks.nets import ResNetFeatures, ViT, resnet18
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#Creating a Transformation Object
transforms = v2.Compose([
    #Converting images to the size that the model expects
    v2.Resize(size=(224,224)),
    v2.ToTensor(), #Converting to tensor
    v2.Normalize(mean = [0.485, 0.456, 0.406],
                                    std = [0.229, 0.224, 0.225]) #Normalizing the data to the data that the ResNet18 was trained on
    
])

# Create custom dataset class
class XrayDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        xray_path = self.df.loc[idx, 'image_path']

        img = Image.open(xray_path).convert("RGB")
        # img = Image.open(xray_path)
        
        label = int(self.df.loc[idx, 'oa_label'])
    
        if self.transforms:
            img = self.transforms(img)

        return img, label
    
cv_best_models = []
cv_best_acc = []
batch_size = 32
epochs = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for fold in range(1, 6):
    xray_train_df = pd.read_csv(f"data/xray_train_cv{fold}.csv")
    xray_valid_df = pd.read_csv(f"data/xray_val_cv{fold}.csv")
    xray_train_dataset = XrayDataset(xray_train_df,  transforms=transforms)
    xray_train_dataloader = DataLoader(xray_train_dataset, batch_size=batch_size, shuffle=True)
    xray_val_dataset = XrayDataset(xray_valid_df, transforms=transforms)
    xray_val_dataloader = DataLoader(xray_val_dataset, batch_size=batch_size)
    
    model = torchvision.models.resnet18(pretrained=True)
    # pretrained_weights = torch.load("../resnet_18_23dataset.pth")
    # model.load_state_dict({k: v for k, v in pretrained_weights.items() if "fc" not in k}, strict=False)
    # model.fc = torch.nn.Linear(in_features=512, out_features=1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = nn.DataParallel(model)
    model = model.cuda()
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # loss_fn = nn.BCEWithLogitsLoss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-04)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    
    best_val_acc = 0.
    best_model = None

    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        losses = []
        correct = 0

        for img, label in tqdm(xray_train_dataloader, leave=False):
            images = img.to(device)
            labels = label.to(device)

            optimizer.zero_grad()
            
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            losses.append(loss.item())
            loss.backward()
    
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            # correct += torch.sum((label.cuda() == (out > 0.5))).item()
            
            optimizer.step()
        print(f"Training Loss: {np.mean(losses)}")
        print(f"Training Accuracy: {correct/(len(xray_train_df))}")
    
        correct = 0
        # losses = 0
        for img, label in tqdm(xray_val_dataloader, leave=False):
            images = img.to(device)
            labels = label.to(device)

            outputs = model(images)
    
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
    
    
        final_acc = correct/len(xray_valid_df)
        print(f"Validation Accuracy: {final_acc}")
    
        if final_acc > best_val_acc:
            best_model = copy.deepcopy(model)
            best_val_acc = final_acc
    
        # scheduler.step()
    
    losses = 0
    correct = 0
    for img, label in tqdm(xray_val_dataloader, leave=False):
        images = img.to(device)
        labels = label.to(device)

        outputs = model(images)

        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)
    
    final_acc = correct/len(xray_valid_df)
    print(f"Fold {fold} Best Model Final Validation Accuracy: {final_acc}")

    cv_best_models.append(copy.deepcopy(best_model))
    cv_best_acc.append(final_acc)


max_acc = np.argmax([item.item() for item in cv_best_acc])
best_model = cv_best_models[max_acc]

xray_test_df= pd.read_csv("data/xray_test.csv")
xray_test_dataset = XrayDataset(xray_test_df, transforms=transforms)
xray_test_dataloader = DataLoader(xray_test_dataset, batch_size=batch_size)

label_test = []
preds_test = []
for img, label in tqdm(xray_test_dataloader):
    images = img.to(device)
    labels = label.to(device)

    outputs = best_model(images)
    _, preds = torch.max(outputs, 1)

    label_test.extend(labels.cpu().numpy())
    preds_test.extend(preds.cpu().numpy())
    

# Calculate metrics
accuracy = accuracy_score(label_test, preds_test)
precision = precision_score(label_test, preds_test)
recall = recall_score(label_test, preds_test)
f1 = f1_score(label_test, preds_test)

# Print results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

PATH = f"/home/feg48/kl_grading_project/xray/model/resnet_best_batch_{batch_size}_lr005.pth"
torch.save(best_model.state_dict(), PATH)
print("save model done!")