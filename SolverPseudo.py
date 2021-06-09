seed = 1
lr = 1e-3
n_epochs = 20
batch_size = 64
size = "512"
n_folds = 5
th = 0.4
loss_name = ["dice", "bce"]
name = "timm-efficientnet-b3"
encode = "noisy-student"
attention = 0
singlefold = 0
foldstop = 0
gpus = "0,1"
fold = 0; epoch = 1

import os

from albumentations.augmentations.transforms import GridDropout
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import re
import cv2
import time
import random
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from apex import amp
import albumentations as A
from segmentation_models_pytorch import Unet, UnetPlusPlus, FPN
from sklearn.model_selection import  KFold, GroupKFold
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from utils.seg.dice import DiceLoss, DiceMetric, SqueezeBCELoss, TverskyLoss
from utils.seg.lovasz import LovaszLoss, StableBCELoss, SymLovaszLoss
from utils.seg.focal import FocalLoss, FocalTverskyLoss
from utils.seg.unetv2 import UnetV2
from utils import make_file_path, setup_seed, Log
modelpath, plotpath, outpath, starttime, basepath = make_file_path(__file__, True)
setup_seed(seed)


image_dir = f"data/{size}/train"
mask_dir = f"data/{size}/masks"
images = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
masks = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir)]

df = pd.DataFrame({"images": images, "masks": masks})
df["patient"] = df.images.apply(lambda x: x.split("/")[-1].split("_")[0])

ps_image_dirs = [f"data/test{size}/train", f"data/external{size}/train"]
ps_mask_dirs = [f"data/test{size}/masks", f"data/external{size}/masks"]
ps_dfs = []
for ps_image_dir, ps_mask_dir in zip(ps_image_dirs, ps_mask_dirs):
    ps_images = [os.path.join(ps_image_dir, f) for f in os.listdir(ps_image_dir)]
    ps_masks = [os.path.join(ps_mask_dir, f) for f in os.listdir(ps_mask_dir)]

    ps_df = pd.DataFrame({"images": ps_images, "masks": ps_masks})
    ps_df["patient"] = ps_df.images.apply(lambda x: x.split("/")[-1].split("_")[0])
    ps_dfs.append(ps_df)
ps_df = pd.concat(ps_dfs).reset_index(drop = True)

# df = pd.concat([df, ps_df]).reset_index(drop = True)


class Data(Dataset):
    def __init__(self, df, trans = None):
        self.df = df
        self.trans = trans
  
    def __getitem__(self, index):
        image = np.array(Image.open(self.df.images.iloc[index]))
        mask = np.array(Image.open(self.df.masks.iloc[index]))
        
        if self.trans is not None:
            aug = self.trans(image = image, mask = mask)
            image = aug["image"]
            mask = aug["mask"]
        image = image.astype(np.float32)
        image = image.transpose(2, 0, 1)
        return image, mask.astype(np.long)
    
    def __len__(self):
        return self.df.shape[0]

class Criterion(nn.Module):
    def __init__(self, loss_name = loss_name, loss_weight = None):
        super(Criterion, self).__init__()
        losses = {
            "bce": SqueezeBCELoss(),
            "dice": DiceLoss(),
            "lovasz": SymLovaszLoss(),
            "focal": FocalLoss(),
            "tver": TverskyLoss(alpha = 0.7, beta = 0.3),
            "focaltver": FocalTverskyLoss()
        }
        self.criterion = [losses[name] for name in loss_name]
        self.loss_weight = loss_weight.astype(float) if loss_weight is not None and len(loss_weight) == len(loss_name) else np.ones(len(loss_name))
        self.loss_weight /= self.loss_weight.sum()

    def forward(self, logits, target):
        loss = 0
        for w, cri in zip(self.loss_weight, self.criterion):
            loss += cri(logits, target) * w
        return loss

class Model(pl.LightningModule):
    def __init__(self, learning_rate = lr):
        super(Model, self).__init__()
        self.learning_rate = learning_rate
        # self.model = FPN(name, encoder_weights = encode)
        self.model = Unet(name, decoder_attention_type = "scse" if attention else None, encoder_weights = encode)
        # self.model = UnetV2(name, decoder_attention_type = "scse" if attention else None, encoder_weights = encode, decoder_act_layer = "frelu")
        # self.model = CSPDarkNet53Unet()
        self.criterion = Criterion(loss_weight = np.array([1., 3.]))

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(model.parameters(), lr = self.learning_rate, weight_decay = 2e-5)
        # self.optimizer = torch.optim.Adam(model.parameters(), lr = self.lr)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = self.learning_rate, steps_per_epoch = int(len(self.train_dataloader())), epochs = n_epochs, anneal_strategy = "linear", final_div_factor = 30,), 'name': 'learning_rate', 'interval':'step', 'frequency': 1}
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = self.criterion(yhat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = self.criterion(yhat, y)
        self.log("val_loss", loss)

        yhat = (yhat.sigmoid() > th).long().squeeze()
        y = y.squeeze()
        inter = (yhat * y).sum()
        card = (yhat + y).sum()

        return loss, inter, card

    def validation_epoch_end(self, outputs):
        inter = torch.stack([x for _, x, _ in outputs]).sum()
        card = torch.stack([x for _, _, x in outputs]).sum()
        dice = 2 * inter / (card + 1e-7)
        self.log("val_metric", dice)


mean = [0.63701425, 0.47097038, 0.68173952]
std = [0.15979014, 0.22442915, 0.14194921]

train_trans = A.Compose([
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.RandomRotate90(),
    # A.Transpose(),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p = 0.9, border_mode=cv2.BORDER_REFLECT),
    A.Blur(p = 0.3),
    A.OneOf([
        A.OpticalDistortion(p = 0.3),
        A.GridDistortion(p = 0.3),
        A.IAAPiecewiseAffine(p = 0.3),
    ], p = 0.7),
    A.OneOf([
        A.HueSaturationValue(25, 40, 30, p = 0.7),
        A.CLAHE(p = 0.7),
        A.RandomBrightnessContrast(0.3, 0.3, p = 0.7),       
    ], p = 0.7),
    A.Normalize(mean = mean, std = std)
], p = 1)
valid_trans = A.Normalize(mean = mean, std = std)

if __name__ == "__main__":
    # split = KFold(n_folds, shuffle = True, random_state = seed)
    split = GroupKFold(n_folds)
    model = None
    for fold in range(fold, n_folds):
        logger = TensorBoardLogger('./logs', name = starttime, version = starttime + "_" + str(fold))
        METRIC = 0.

        train_idx, valid_idx = list(split.split(df, groups = df.patient))[fold]
        df_train = df.iloc[train_idx] if not singlefold else df.copy()
        df_valid = df.iloc[valid_idx]

        df_train = pd.concat([df_train, ps_df])

        ds_train = Data(df_train, train_trans)
        ds_valid = Data(df_valid, valid_trans)
        dl_train = DataLoader(ds_train, batch_size = batch_size, shuffle = True, num_workers = 4, drop_last = False)
        dl_valid = DataLoader(ds_valid, batch_size = batch_size, shuffle = False, num_workers = 4)

        callback = pl.callbacks.ModelCheckpoint(
            filename = starttime + "_" + str(fold) + '_{epoch}_{val_metric:.3f}',
            save_last = True,
            mode = "max",
            monitor = 'val_metric'
        )

        model = Model()
        # model = model.load_from_checkpoint(f"./logs/Mar291351/Mar291351_{fold}/checkpoints/last.ckpt")
        # model.model.load_state_dict(model.load_from_checkpoint(f"logs/Mar182219_{fold}/checkpoints/last.ckpt").model.state_dict())
        trainer = pl.Trainer(
            gpus = len(gpus.split(",")), 
            precision = 16, amp_backend = "native", amp_level = "O1", 
            accelerator = "dp",
            gradient_clip_val = 0.5,
            max_epochs = n_epochs,
            stochastic_weight_avg = True,
            logger = logger,
            progress_bar_refresh_rate = 0,
            callbacks = [callback]
        )
        trainer.fit(model, dl_train, dl_valid)

        if foldstop or singlefold: break
