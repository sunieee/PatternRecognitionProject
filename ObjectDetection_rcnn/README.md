## oriented RCNN

evaluation: https://github.com/CAPTAIN-WHU/DOTA_devkit/tree/master

mmrotate DOC: https://mmrotate.readthedocs.io/en/latest/tutorials/customize_dataset.html

install mmrotate (DO NOT use pip): https://github.com/open-mmlab/mmrotate

train: 

- 训练自训练模型：`./train.sh`

- 训练迁移模型：`./tuning.sh`

inference & test: 

- 官方预模型：`./test.sh`

Task:

1. 比较使用官方预模型（pretrain）、自训练模型（train）、迁移训练模型（transfer）
2. 比较不同模型类型：oriented_rcnn, rotated_faster_rcnn, rotated_retinanet 模型
3. 对最佳模型错误样例进行案例分析

### 官方预训练模型 (pretrain)

![image-20240612122756609](https://n.ye-sun.com/gallery/2024/202406121228326.png)

![image-20240612124054632](https://n.ye-sun.com/gallery/2024/202406121240823.png)

### 自训练模型 (train)

修改过时的代码：删除sampler，否则len(data_loader)为0，添加device为整数情况

![image-20240612163434314](https://n.ye-sun.com/gallery/2024/202406121636313.png)

开始训练：

![image-20240612163621848](https://n.ye-sun.com/gallery/2024/202406121636157.png)

![output (2)](https://n.ye-sun.com/gallery/2024/202406121804157.png)

![image-20240612165150987](https://n.ye-sun.com/gallery/2024/202406121651342.png)

![image-20240612165136592](https://n.ye-sun.com/gallery/2024/202406121651746.png)

### 微调模型 (tuning)

预训练模型有15个类别（不含background），但是目标探测任务只有1类，因此需要进行调整，在加载模型时只取模型前面层的参数，而忽略最后一层参数

```shell
2024-06-12 16:55:11,849 - mmrotate - INFO - Set random seed to 502270266, deterministic: False
2024-06-12 16:55:12,695 - mmrotate - INFO - load checkpoint from local path: oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth
2024-06-12 16:55:12,837 - mmrotate - WARNING - The model and loaded state dict do not match exactly

size mismatch for roi_head.bbox_head.fc_cls.weight: copying a param with shape torch.Size([16, 1024]) from checkpoint, the shape in current model is torch.Size([2, 1024]).
size mismatch for roi_head.bbox_head.fc_cls.bias: copying a param with shape torch.Size([16]) from checkpoint, the shape in current model is torch.Size([2]).
2024-06-12 16:55:12,842 - mmrotate - INFO - resumed epoch 12, iter 76800
2024-06-12 16:55:12,843 - mmrotate - INFO - Start running, host: sy@vis, work_dir: /home/sy/2024/PR/ObjectDetection/work_dirs/transfer
```

![image-20240612171745695](https://n.ye-sun.com/gallery/2024/202406121717151.png)

![output (3)](https://n.ye-sun.com/gallery/2024/202406121811326.png)

![image-20240612183518050](https://n.ye-sun.com/gallery/2024/202406121845715.png)

![image-20240612183535710](https://n.ye-sun.com/gallery/2024/202406121845323.png)

### 模型对比

![image-20240612184453532](https://n.ye-sun.com/gallery/2024/202406121844222.png)

| Task     | Model               | AP    |
| -------- | ------------------- | ----- |
| pretrain | oriented_rcnn       | 95.59 |
| train    | oriented_rcnn       | 88.65 |
| tuning   | oriented_rcnn       | 95.18 |
| pretrain | rotated_faster_rcnn | 93.03 |
| pretrain | rotated_retinanet   | 78.54 |