import os
import torch
from torch.utils.data import Dataset
from PIL import Image

from torchvision import transforms
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

class KSHImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: 이미지가 들어있는 최상위 폴더
        transform: 이미지 전처리 도구
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths= []
        self.labels = []

        self.class_to_idx = {"cat":0, "dog":1}

        for label_name in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label_name)
            if os.path.isdir(label_path):
                for img_name in os.listdir(label_path):
                    self.image_paths.append(os.path.join(label_path, img_name))
                    self.labels.append(self.class_to_idx[label_name])

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

class CNNNet(nn.Module):
    def __init__(self, num_classes = 2):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #Drop out 추가...
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = self.dropout1(x)
        
        # Flatten: 1차원으로 펼치기
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        
        x = self.dropout2(x)
        
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
lr = 0.001
epochs = 10

kshtransform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),])
train_data = KSHImageDataset(root_dir='/workspace/data/CnD/train', transform=kshtransform)
val_data = KSHImageDataset(root_dir='/workspace/data/CnD/val', transform=kshtransform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

# for images, labels in train_loader:
#     print(images.shape)
#     break

model = CNNNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    # --- 학습 모드 ---
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
    
    # --- 평가 모드 (Eval) ---
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            
    # 에폭별 결과 출력
    epoch_loss = train_loss / len(train_loader.dataset)
    epoch_acc = correct.double() / len(val_loader.dataset)
    
    print(f'Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Val Acc: {epoch_acc:.4f}')

print("학습 및 검증 완료!")