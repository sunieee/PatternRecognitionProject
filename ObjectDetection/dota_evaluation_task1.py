import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import polyiou
from functools import partial

def parse_args():
    parser = argparse.ArgumentParser(description='DOTA evaluation script')
    parser.add_argument('--detpath', type=str, required=True, help='Path to the detection results')
    parser.add_argument('--annopath', type=str, required=True, help='Path to the ground truth annotations')
    parser.add_argument('--classname', type=str, required=True, help='Class to evaluate')
    parser.add_argument('--ovthresh', type=float, default=0.5, help='Overlap threshold')
    args = parser.parse_args()
    return args

def parse_gt(filename):
    objects = []
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if line:
                splitlines = line.strip().split(' ')
                object_struct = {}
                if len(splitlines) < 9:
                    continue
                object_struct['name'] = splitlines[8]
                object_struct['difficult'] = 0 if len(splitlines) == 9 else int(splitlines[9])
                object_struct['bbox'] = [float(splitlines[i]) for i in range(8)]
                objects.append(object_struct)
            else:
                break
    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap, mpre, mrec

def voc_eval(detpath, annopath, classname, ovthresh=0.5, use_07_metric=False):
    imagenames = [os.path.splitext(f)[0] for f in os.listdir(annopath) if f.endswith('.txt')]
    recs = {imagename: parse_gt(os.path.join(annopath, imagename + '.txt')) for imagename in imagenames}

    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox, 'difficult': difficult, 'det': det}

    image_ids = []
    confidence = []
    BB = []

    for imagename in imagenames:
        detfile = os.path.join(detpath, f"{imagename}.txt")
        if not os.path.exists(detfile):
            print(f"Detection file {detfile} does not exist, skipping.")
            continue
        with open(detfile, 'r') as f:
            lines = f.readlines()
        splitlines = [x.strip().split(' ') for x in lines]
        image_ids.extend([imagename] * len(splitlines))
        confidence.extend([float(x[9]) for x in splitlines])
        BB.extend([[float(z) for z in x[:8]] for x in splitlines])

    confidence = np.array(confidence)
    BB = np.array(BB)

    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            BBGT_xmin = np.min(BBGT[:, 0::2], axis=1)
            BBGT_ymin = np.min(BBGT[:, 1::2], axis=1)
            BBGT_xmax = np.max(BBGT[:, 0::2], axis=1)
            BBGT_ymax = np.max(BBGT[:, 1::2], axis=1)
            bb_xmin = np.min(bb[0::2])
            bb_ymin = np.min(bb[1::2])
            bb_xmax = np.max(bb[0::2])
            bb_ymax = np.max(bb[1::2])

            ixmin = np.maximum(BBGT_xmin, bb_xmin)
            iymin = np.maximum(BBGT_ymin, bb_ymin)
            ixmax = np.minimum(BBGT_xmax, bb_xmax)
            iymax = np.minimum(BBGT_ymax, bb_ymax)
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            uni = ((bb_xmax - bb_xmin + 1.) * (bb_ymax - bb_ymin + 1.) +
                   (BBGT_xmax - BBGT_xmin + 1.) *
                   (BBGT_ymax - BBGT_ymin + 1.) - inters)

            overlaps = inters / uni

            BBGT_keep_mask = overlaps > 0
            BBGT_keep = BBGT[BBGT_keep_mask, :]
            BBGT_keep_index = np.where(overlaps > 0)[0]

            def calcoverlaps(BBGT_keep, bb):
                overlaps = []
                for index, GT in enumerate(BBGT_keep):
                    overlap = polyiou.iou_poly(polyiou.VectorDouble(BBGT_keep[index]), polyiou.VectorDouble(bb))
                    overlaps.append(overlap)
                return overlaps

            if len(BBGT_keep) > 0:
                overlaps = calcoverlaps(BBGT_keep, bb)
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                jmax = BBGT_keep_index[jmax]

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

        print(f"Image: {image_ids[d]}, Confidence: {confidence[d]}, TP: {tp[d]}, FP: {fp[d]}")

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap, prec, rec = voc_ap(rec, prec, use_07_metric)
    return rec, prec, ap

def main():
    args = parse_args()
    detpath = args.detpath
    annopath = args.annopath
    classname = args.classname
    ovthresh = args.ovthresh

    rec, prec, ap = voc_eval(detpath, annopath, classname, ovthresh)
    print(f'Recall: {rec}')
    print(f'Precision: {prec}')
    print(f'AP: {ap}')

    # Plot Precision-Recall curve
    plt.figure()
    plt.plot(rec, prec, lw=2, label=f'Precision-Recall curve (AP={ap:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall curve: {classname}')
    plt.legend(loc="best")
    plt.savefig('test.png')

if __name__ == '__main__':
    main()
