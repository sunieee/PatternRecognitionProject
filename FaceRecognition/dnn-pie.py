import os
import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
from torch import nn, optim
from torchvision import transforms, models
from collections import defaultdict
from PIL import Image

class PIEMatDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.image_types = []
        self.data = []
        self.labels = []
        self.image_types = []
        self.load_mat_files()

    def load_mat_files(self):
        for mat_file in os.listdir(self.root_dir):
            if mat_file.endswith('.mat'):
                mat_path = os.path.join(self.root_dir, mat_file)
                mat_data = scipy.io.loadmat(mat_path)
                images = mat_data['fea']
                labels = mat_data['gnd'].flatten()
                image_type = mat_file.split('_')[0]
                
                self.data.append(images)
                self.labels.append(labels)
                self.image_types.extend([image_type] * len(labels))
        
        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx].reshape(64, 64).astype(np.uint8)
        label = self.labels[idx] - 1  # Assuming labels start from 1
        image_type = self.image_types[idx]
        
        image = np.stack([image] * 3, axis=-1)  # Convert to RGB by stacking the gray image three times
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, image_type

# 定义数据变换
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 实例化数据集
dataset = PIEMatDataset(root_dir="data/PIE dataset", transform=transform)

type2count = defaultdict(int)
type2labels = defaultdict(list)
for ix, t in enumerate(dataset.image_types):
    type2count[t] += 1
    type2labels[t].append(dataset.labels[ix])
type2labels = {k: len(set(v)) for k, v in type2labels.items()}

print('type2count', type2count)
print('type2labels', type2labels)

# 划分数据集，按 70%:10%:20% 划分
train_size = int(0.7 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# 使用DataLoader加载数据
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(np.unique(dataset.labels)))  # 调整最后一层以适应我们的数据集

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 添加 Early Stopping 机制
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), 'checkpoint.pt')

early_stopping = EarlyStopping(patience=5)

# 训练模型
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)
        
        # 每个epoch有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = dataloaders['train']
            else:
                model.eval()
                dataloader = dataloaders['val']
            
            running_loss = 0.0
            running_corrects = 0
            
            # 遍历数据
            for inputs, labels, _ in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # 前向传播
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # 反向传播和优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            
            # Early stopping 逻辑
            if phase == 'val':
                early_stopping(epoch_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    model.load_state_dict(torch.load('checkpoint.pt'))
                    return model
    
    return model

dataloaders = {
    'train': train_dataloader,
    'val': val_dataloader
}

model = train_model(model, dataloaders, criterion, optimizer, num_epochs=25)

# 评估模型
def evaluate_model(model, dataloader):
    model.eval()
    running_corrects = defaultdict(int)
    total_counts = defaultdict(int)
    
    for inputs, labels, image_types in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
        
        for label, pred, image_type in zip(labels, preds, image_types):
            if label == pred:
                running_corrects[image_type] += 1
            total_counts[image_type] += 1
    
    print("| Image Type | Accuracy | # Corrects | # Total |")
    print("|------------|----------|------------|---------|")
    total_corrects = 0
    total_images = 0
    for image_type in total_counts:
        accuracy = running_corrects[image_type] / total_counts[image_type]
        total_corrects += running_corrects[image_type]
        total_images += total_counts[image_type]
        print(f"| {image_type:<10} | {accuracy:.4f}   | {running_corrects[image_type]:<10} | {total_counts[image_type]:<7} |")
    
    total_accuracy = total_corrects / total_images
    print(f"| **Total**  | {total_accuracy:.4f}   | {total_corrects:<10} | {total_images:<7} |")

evaluate_model(model, test_dataloader)
