
export PYTHONPATH=$PYTHONPATH:./DOTA_devkit

# python process_train_val.py

python train.py --config oriented_rcnn_r50_fpn_1x_dota_le90.py --work-dir work_dirs/tuning \
    --no-validate --resume-from oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth --auto-resume

python inference.py --output-label-dir results/tuninglabel/ --output-image-dir results/tuningimage/ --checkpoint work_dirs/tuning/latest.pth

python dota_evaluation_task1.py --detpath results/tuninglabel/ --annopath data/test/labelTxt/ \
    --classname Airplane --ovthresh 0.5