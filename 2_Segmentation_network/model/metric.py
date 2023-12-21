from utils import *


def get_iou(tp, fp, fn, tn):
    return iou_score(tp, fp, fn, tn, reduction="micro-imagewise")


def get_accuracy(tp, fp, fn, tn):
    return accuracy(tp, fp, fn, tn, reduction="micro-imagewise")


def get_f1_score(tp, fp, fn, tn):
    return f1_score(tp, fp, fn, tn, reduction="micro-imagewise")


def get_precision(tp, fp, fn, tn):
    return precision(tp, fp, fn, tn, reduction="micro-imagewise")
