
export PYTHONPATH=$PYTHONPATH:./DOTA_devkit

# python process_train_val.py

python train.py --config oriented_rcnn_r50_fpn_1x_dota_le90.py --work-dir work_dirs/train \
    --no-validate

python inference.py --output-label-dir results/trainlabel/ --output-image-dir results/trainimage/ --checkpoint work_dirs/train/latest.pth

python dota_evaluation_task1.py --detpath results/trainlabel/ --annopath data/test/labelTxt/ \
            --classname Airplane --ovthresh 0.5