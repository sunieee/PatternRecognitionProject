import torch
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import os
from torch import nn
from torchvision import transforms
from PIL import Image

dataset = 'fr'
classes = 338

# 获取中间层输出的钩子
class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []

# 可视化特征图
def visualize_feature_maps(features, layer_name, save_dir='feature_maps'):
    os.makedirs(save_dir, exist_ok=True)
    features = features.detach().cpu().numpy()
    num_feature_maps = features.shape[1]

    fig, axes = plt.subplots(1, num_feature_maps, figsize=(num_feature_maps, 1))
    for i in range(num_feature_maps):
        feature_map = features[0, i, :, :]
        axes[i].imshow(feature_map, cmap='gray')
        axes[i].axis('off')

    plt.savefig(os.path.join(save_dir, f"{layer_name}_feature_maps.png"))
    plt.close(fig)

# 加载训练好的模型
model_path = 'checkpoint-fr.pt'
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, classes)
model.load_state_dict(torch.load(model_path))
model.eval()

# 定义数据变换
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载一个示例图像
image_path = './data/faces96/9540628/9540628.8.jpg'  # 请替换为实际图像路径
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(next(model.parameters()).device)

# 注册钩子以保存中间层输出
save_output = SaveOutput()
hooks = []
layers_to_hook = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']

for layer_name in layers_to_hook:
    layer = dict(model.named_children())[layer_name]
    if isinstance(layer, nn.Sequential):
        for sub_layer in layer:
            hooks.append(sub_layer.register_forward_hook(save_output))
    else:
        hooks.append(layer.register_forward_hook(save_output))

# 通过模型前向传播图像
with torch.no_grad():
    model(image)

# 可视化并保存中间层特征图
for idx, layer_name in enumerate(layers_to_hook):
    visualize_feature_maps(save_output.outputs[idx], layer_name)

# 清除钩子
for hook in hooks:
    hook.remove()
