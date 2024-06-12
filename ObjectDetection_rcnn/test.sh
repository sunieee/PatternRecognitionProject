export PYTHONPATH=$PYTHONPATH:./DOTA_devkit

python inference.py --output-label-dir results/pretrainlabel/ --output-image-dir results/pretrainimages/ \
            --config 
python dota_evaluation_task1.py --detpath results/pretrainlabel/ --annopath data/test/labelTxt/ \
            --classname Airplane --ovthresh 0.5

# python inference.py --output-label-dir results/rotated_faster_rcnnlabel/ --output-image-dir results/rotated_faster_rcnnimages/ \
#             --config rotated_faster_rcnn_r50_fpn_1x_dota_le90.py --checkpoint rotated_faster_rcnn_r50_fpn_1x_dota_le90-0393aa5c.pth
# python dota_evaluation_task1.py --detpath results/rotated_faster_rcnnlabel/ --annopath data/test/labelTxt/ \
#             --classname Airplane --ovthresh 0.5

# python inference.py --output-label-dir results/rotated_retinanetlabel/ --output-image-dir results/rotated_retinanetimages/ \
#             --config rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90.py --checkpoint rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90-1da1ec9c.pth
# python dota_evaluation_task1.py --detpath results/rotated_retinanetlabel/ --annopath data/test/labelTxt/ \
#             --classname Airplane --ovthresh 0.5