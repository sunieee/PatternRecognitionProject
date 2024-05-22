import os
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import shutil
from torch import nn, optim
from torchvision import transforms, models
from collections import defaultdict

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = []
        self.image_types = []
        
        # 读取所有子文件夹和图像
        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if subdir in ['faces95', 'faces96', 'grimace']:
                self.read_subdir(subdir_path, subdir)
            elif subdir == 'faces94':
                for f in os.listdir(subdir_path):
                    self.read_subdir(os.path.join(subdir_path, f), subdir)
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.labels = [self.class_to_idx[label] for label in self.labels]

    def read_subdir(self, subdir_path, subdir_name):
        for person in os.listdir(subdir_path):
            person_path = os.path.join(subdir_path, person)
            if os.path.isdir(person_path):
                images = sorted(os.listdir(person_path))
                for img_name in images:
                    img_path = os.path.join(person_path, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(person)
                    self.image_types.append(subdir_name)
                    if person not in self.classes:
                        self.classes.append(person)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, self.image_types[idx]

# 定义数据变换
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 实例化数据集
dataset = FaceDataset(root_dir="data", transform=transform)

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
model.fc = nn.Linear(num_ftrs, len(dataset.classes))  # 调整最后一层以适应我们的数据集

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
    
    for image_type in total_counts:
        accuracy = running_corrects[image_type] / total_counts[image_type]
        print(f"Accuracy for {image_type}: {accuracy:.4f}")
        print(running_corrects[image_type], total_counts[image_type])

    accuracy = sum(running_corrects.values()) / sum(total_counts.values())
    print(f"Total Accuracy: {accuracy:.4f}")

evaluate_model(model, test_dataloader)
