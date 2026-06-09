"""Utility helpers used by the MiT referring-segmentation pipeline."""
import argparse
import logging

import torch


# --------------------------- argparse type helpers ---------------------------
def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected." + v)


def str2bools(v):
    return list(map(str2bool, v.split("-")))


def str2ints(v):
    return list(map(int, v.split("-")))


# ------------------------------- logging -------------------------------------
def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s: %(message)s"))
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)


def log_message(message, rank):
    # Only the main process (rank 0) writes logs.
    if rank != 0:
        return
    logging.log(msg=message, level=logging.DEBUG)


def format_string(text: str):
    return text[0:1].upper() + text[1:]


# ----------------------------- segmentation IoU ------------------------------
def compute_IoU_2class_batch(pred, gt):
    # pred: [bs, 2, h, w], gt: [bs, h, w]
    pred = pred.argmax(1)
    intersection = torch.sum(torch.mul(pred, gt), dim=(1, 2))
    union = torch.sum(torch.add(pred, gt), dim=(1, 2)) - intersection
    iou = intersection / union
    iou = torch.nan_to_num(iou, nan=0, posinf=0, neginf=0)
    return iou, intersection, union


def compute_IoU_1class_batch(pred, gt):
    # pred: [bs, 1, h, w], gt: [bs, h, w]
    pred = torch.sigmoid(pred).squeeze(1)
    pred = (pred > 0.5).long()
    intersection = torch.sum(torch.mul(pred, gt), dim=(1, 2))
    union = torch.sum(torch.add(pred, gt), dim=(1, 2)) - intersection
    iou = intersection / union
    iou = torch.nan_to_num(iou, nan=0, posinf=0, neginf=0)
    return iou, intersection, union


def compute_dice_from_iou(iou):
    dice = (2 * iou) / (1 + iou)
    return dice
