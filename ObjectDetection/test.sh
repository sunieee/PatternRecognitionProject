export PYTHONPATH=$PYTHONPATH:./DOTA_devkit

# python inference.py

python dota_evaluation_task1.py --detpath results/pretrain/label/ --annopath data/test/labelTxt/ \
            --classname Airplane --ovthresh 0.5