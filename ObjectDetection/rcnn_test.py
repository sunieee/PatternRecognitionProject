from mmdet.apis import init_detector, inference_detector
import mmrotate
import mmcv
import torch

config_file = 'oriented_rcnn_r50_fpn_1x_dota_le90.py'
checkpoint_file = 'oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth'
device = 'cuda:0'

model = init_detector(config_file, checkpoint_file, device=device)

# Load image
img = 'data/test/images/5__1__0___0.png'
# img = mmcv.imread(img)
# img = torch.from_numpy(img).to(device)

# Perform inference
result = inference_detector(model, img)

print(result)
print('plane class:', result[0])

# Visualize results
out_file = 'rcnn_test.png'
model.show_result(img, result, out_file=out_file)
print(f'Result saved to {out_file}')