from os.path import join, isfile, isdir
from os import listdir
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from argparse import ArgumentParser

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

import torch
from tqdm import tqdm

from options.train_options import TrainOptions

cfg = get_cfg()
cfg.MODEL.DEVICE = "cpu"
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

opt = TrainOptions().parse()
dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
dir_Amask = os.path.join(opt.dataroot, opt.phase + 'Amask')
dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
dir_Bmask = os.path.join(opt.dataroot, opt.phase + 'Bmask')

image_listA = [f for f in listdir(dir_A) if isfile(join(dir_A, f))]
image_listB = [f for f in listdir(dir_B) if isfile(join(dir_B, f))]

if os.path.isdir(dir_Amask) is False:
    os.makedirs(dir_Amask)
if os.path.isdir(dir_Bmask) is False:
    os.makedirs(dir_Bmask)

for image_path in tqdm(image_listA):
    if isfile(join(dir_Amask, image_path)):
        continue
    img = cv2.imread(join(dir_A, image_path))
    outputs = predictor(img)
    pred_masks = outputs["instances"].pred_masks.cpu().data.numpy()
    masked_image = np.zeros((img.shape[0], img.shape[1]))
    for c in range(pred_masks.shape[0]):
        masked_image = np.add(masked_image, pred_masks[c, :, :]*255)
    masked_image = np.ones((img.shape[0], img.shape[1]))*255 - masked_image
    # masked_image = img + masked_image
    save_path = join(dir_Amask, image_path)
    cv2.imwrite(save_path, masked_image)

for image_path in tqdm(image_listB):
    if isfile(join(dir_Bmask, image_path)):
        continue
    img = cv2.imread(join(dir_B, image_path))
    outputs = predictor(img)
    pred_masks = outputs["instances"].pred_masks.cpu().data.numpy()
    masked_image = np.zeros((img.shape[0], img.shape[1]))
    for c in range(pred_masks.shape[0]):
        masked_image = np.add(masked_image, pred_masks[c, :, :]*255)
    masked_image = np.ones((img.shape[0], img.shape[1]))*255 - masked_image
    # masked_image = img + masked_image
    save_path = join(dir_Bmask, image_path)
    cv2.imwrite(save_path, masked_image)