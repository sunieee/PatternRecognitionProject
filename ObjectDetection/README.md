## oriented RCNN

evaluation: https://github.com/CAPTAIN-WHU/DOTA_devkit/tree/master

mmrotate DOC: https://mmrotate.readthedocs.io/en/latest/tutorials/customize_dataset.html

install mmrotate (DONOT use pip): https://github.com/open-mmlab/mmrotate

train: 
- 训练自训练模型：`./train.sh`
- 训练迁移模型：`./transfer.sh`

inference & test: 
- 官方预模型：`./test.sh`

Task:
1. 比较使用官方预模型、自训练模型、训练迁移模型
2. 对错误样例进行案例分析