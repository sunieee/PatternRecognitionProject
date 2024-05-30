import torch
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import os
from torch import nn

dataset = 'fr'
classes = 338

# 可视化卷积层参数
def visualize_conv_layer(layer, layer_name, save_dir='conv_visualizations'):
    os.makedirs(save_dir, exist_ok=True)
    
    if isinstance(layer, nn.Conv2d):
        weights = layer.weight.data.cpu().numpy()
        num_kernels = weights.shape[0]
        
        fig, axes = plt.subplots(1, num_kernels, figsize=(num_kernels, 1))
        for i in range(num_kernels):
            kernel = weights[i, 0, :, :]
            axes[i].imshow(kernel, cmap='gray')
            axes[i].axis('off')
        
        plt.savefig(os.path.join(save_dir, f"{layer_name}_kernels.png"))
        plt.close(fig)
    else:
        print(f"Layer {layer_name} is not a Conv2d layer.")

# 可视化并比较不同卷积深度的参数
def compare_conv_layers(model, layers, save_dir='conv_comparisons'):
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(len(layers), 1, figsize=(10, len(layers) * 5))
    for idx, (layer_name, layer) in enumerate(layers.items()):
        if isinstance(layer, nn.Conv2d):
            weights = layer.weight.data.cpu().numpy()
            num_kernels = weights.shape[0]
            for i in range(num_kernels):
                kernel = weights[i, 0, :, :]
                axes[idx].imshow(kernel, cmap='gray')
                axes[idx].set_title(f"{layer_name} kernel {i}")
                axes[idx].axis('off')
        else:
            print(f"Layer {layer_name} is not a Conv2d layer.")
    
    plt.savefig(os.path.join(save_dir, "conv_layers_comparison.png"))
    plt.close(fig)

# 加载训练好的模型
model_path = f'checkpoint-{dataset}.pt'
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, classes)
model.load_state_dict(torch.load(model_path))
model.eval()

# 可视化并比较卷积层
layers_to_visualize = {'conv1': model.conv1}
residual_blocks = {'layer1': model.layer1, 'layer2': model.layer2, 'layer3': model.layer3, 'layer4': model.layer4}

# 可视化第一个卷积层
visualize_conv_layer(model.conv1, 'conv1')

# 可视化残差块中的卷积层
for block_name, block in residual_blocks.items():
    for i, layer in enumerate(block):
        visualize_conv_layer(layer.conv1, f'{block_name}_conv1_{i}')

# 比较不同卷积深度的参数
layers_to_compare = {
    'conv1': model.conv1,
    'layer1_conv1': model.layer1[0].conv1,
    'layer2_conv1': model.layer2[0].conv1,
    'layer3_conv1': model.layer3[0].conv1,
    'layer4_conv1': model.layer4[0].conv1
}
compare_conv_layers(model, layers_to_compare)
