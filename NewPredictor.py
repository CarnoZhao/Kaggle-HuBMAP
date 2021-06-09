sz = 512  # the size of tiles
reduce = 1024 // sz  # reduce the original images by 4 times
TH = 0.45  # threshold for positive predictions
bs = 64
DATA = './data/raw/test'
csv = 'data/raw/sample_submission.csv'
suffix = "090947"
import os
import numpy as np
from segmentation_models_pytorch.unet import decoder
MODELS = [
    # *[(_, "timm-regnety_032") for _ in os.popen(f'find ./logs/ -name "*epoch*.ckpt" | grep {suffix.split("_")[0]}').read().split()],
    *[(_, "timm-efficientnet-b3") for _ in os.popen(f'find ./logs/ -name "*epoch*.ckpt" | grep {suffix.split("_")[0]}').read().split()],
]
# name = "timm-efficientnet-b2"
mean = np.array([0.63701425, 0.47097038, 0.68173952])
std = np.array([0.15979014, 0.22442915, 0.14194921])

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import gc
import cv2
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tifffile as tiff
from tqdm import tqdm
import rasterio
from rasterio.windows import Window
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from segmentation_models_pytorch import Unet, UnetPlusPlus
from utils.seg.cspdarnet53 import CSPDarkNet53Unet
from utils.seg.unetv2 import UnetV2
import pytorch_lightning as pl

df_sample = pd.read_csv(csv)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

s_th = 40 
p_th = 1000 * (sz // 256) ** 2
identity = rasterio.Affine(1, 0, 0, 0, 1, 0)

def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))

def rle_encode_less_memory(img):
    pixels = img.T.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def enc2mask(encs, shape):
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for m,enc in enumerate(encs):
        if isinstance(enc,np.float) and np.isnan(enc): continue
        s = enc.split()
        for i in range(len(s)//2):
            start = int(s[2*i]) - 1
            length = int(s[2*i+1])
            img[start:start+length] = 1 + m
    return img.reshape(shape).T

class HuBMAPDataset(Dataset):
    def __init__(self, idx, sz=sz, reduce=reduce):
        self.data = rasterio.open(os.path.join(DATA,idx+'.tiff'), transform = identity,num_threads='all_cpus')
        if self.data.count != 3:
            self.layers = [rasterio.open(subd) for subd in self.data.subdatasets]
        self.shape = list(self.data.shape)
        self.reduce = reduce
        self.sz = (reduce * sz) // 2
        self.pad0 = (self.sz - self.shape[0] % self.sz) % self.sz
        self.pad1 = (self.sz - self.shape[1] % self.sz) % self.sz
        self.n0max = (self.shape[0] + self.pad0) // self.sz - 1
        self.n1max = (self.shape[1] + self.pad1) // self.sz - 1

    def __len__(self):
        return self.n0max * self.n1max

    def __getitem__(self, idx):
        n0, n1 = idx // self.n1max, idx % self.n1max
        x0, y0 = -self.pad0 // 2 + n0 * self.sz, -self.pad1 // 2 + n1 * self.sz
        p00, p01 = max(0, x0), min(x0 + self.sz * 2, self.shape[0])
        p10, p11 = max(0, y0), min(y0 + self.sz * 2, self.shape[1])
        img = np.zeros((self.sz * 2, self.sz * 2, 3), np.uint8)
        xs = (p00, p01)
        ys = (p10, p11)
        if self.data.count == 3:
            image = self.data.read([1, 2, 3], window = Window.from_slices(xs, ys))
            image = np.moveaxis(image, 0, -1)
        else:
            image = np.zeros((p01 - p00, p11 - p10, 3), dtype = np.uint8)
            for fl in range(3):
                image[:,:,fl] = self.layers[fl].read(window = Window.from_slices(xs, ys))

        img[(p00 - x0):(p01 - x0), (p10 - y0):(p11 - y0)] = image
        if self.reduce != 1:
            img = cv2.resize(img, (self.sz * 2 // reduce, self.sz * 2 // reduce), interpolation = cv2.INTER_AREA)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, s, _ = cv2.split(hsv)
        if (s > s_th).sum() <= p_th or img.sum() <= p_th:
            return img2tensor((img / 255.0 - mean) / std), -1
        else:
            return img2tensor((img / 255.0 - mean) / std), idx

class Model_pred:
    def __init__(self, models, dl, tta: bool = True, half: bool = False):
        self.models = models
        self.dl = dl
        self.tta = tta
        self.half = half # 2 classes, width first then height

    def __iter__(self):
        count = 0
        with torch.no_grad():
            for x, y in tqdm(iter(self.dl), total = len(self.dl)):
                if ((y >= 0).sum() > 0):
                    x = x[y >= 0].to(device)
                    y = y[y >= 0]
                    py = None
                    for model in self.models:
                        p = model(x)
                        p = torch.sigmoid(p).detach()
                        if py is None:
                            py = p
                        else:
                            py += p
                    if self.tta:
                        flips = [[-1], [-2], [-2, -1]]
                        for f in flips:
                            xf = torch.flip(x, f)
                            for model in self.models:
                                p = model(xf)
                                p = torch.flip(p, f)
                                p = torch.sigmoid(p)
                                py += p.detach()
                        py /= (1+len(flips))
                    py /= len(self.models)
                    py = F.upsample(py, scale_factor = reduce, mode="bilinear")
                    py = py.permute(0, 2, 3, 1).float().cpu()
                    batch_size = len(py)
                    for i in range(batch_size):
                        yield py[i], y[i]
                        count += 1

    def __len__(self):
        return len(self.dl.dataset)

class Model(pl.LightningModule):
    def __init__(self, learning_rate = 1e-3):
        super(Model, self).__init__()
        self.learning_rate = learning_rate
        self.model = Unet(name, decoder_attention_type = None, encoder_weights = None, classes = 1)
        # self.model = CSPDarkNet53Unet()
        self.criterion = None

    def forward(self, x):
        return self.model(x)


_models = []
for path, name in MODELS:
    model = Model()
    model = model.load_from_checkpoint(path)
    model.float()
    model.eval()
    model.to(device)
    _models.append(model)

for model_idx, model in enumerate(_models):
    models = _models # [model]
    names, preds = [], []
    fps = []
    fns = []
    tps = []
    for idx, row in tqdm(df_sample.iterrows(), total = len(df_sample)):
        idx = row['id']
        ds = HuBMAPDataset(idx)
        dl = DataLoader(ds, bs, num_workers = 0, shuffle = False, pin_memory = True)
        mp = Model_pred(models, dl)
        mask = torch.zeros((ds.n0max + 1) * ds.sz, (ds.n1max + 1) * ds.sz, dtype = torch.int8)
        line = torch.zeros(ds.sz * 2, (ds.n1max + 1) * ds.sz, dtype = torch.float32)
        pre_n0 = 0
        for p, i in iter(mp):
            n0, n1 = i // ds.n1max, i % ds.n1max
            if n0 != pre_n0:
                line[:ds.sz,:ds.sz] /= 1 + (pre_n0 != 0)
                line[:ds.sz,-ds.sz:] /= 1 + (pre_n0 != 0)
                line[:ds.sz,ds.sz:-ds.sz] /= 2 + 2 * (pre_n0 != 0)
                mask[pre_n0 * ds.sz:pre_n0 * ds.sz + ds.sz] = (line[:ds.sz] > TH).type(torch.int8)
                line[:ds.sz] = line[-ds.sz:]
                line[-ds.sz:] = 0
            line[:,n1 * ds.sz:n1 * ds.sz + ds.sz * 2] += p.squeeze(-1)
            pre_n0 = n0
        line[:ds.sz,:ds.sz] /= 1 + (pre_n0 != ds.n0max)
        line[:ds.sz,-ds.sz:] /= 1 + (pre_n0 != ds.n0max)
        line[:ds.sz,ds.sz:-ds.sz] /= 2 + 2 * (pre_n0 != ds.n0max)
        line[-ds.sz:,:ds.sz] /= 1 + (pre_n0 + 1 != ds.n0max)
        line[-ds.sz:,-ds.sz:] /= 1 + (pre_n0 + 1 != ds.n0max)
        line[-ds.sz:,ds.sz:-ds.sz] /= 2 + 2 * (pre_n0 + 1 != ds.n0max)
        mask[pre_n0 * ds.sz:pre_n0 * ds.sz + 2 * ds.sz] = (line > TH).type(torch.int8)
        mask = mask[ds.pad0 // 2:-(ds.pad0 - ds.pad0 // 2) if ds.pad0 > 0 else (ds.n0max + 1) * ds.sz, ds.pad1 // 2:-(ds.pad1 - ds.pad1 // 2) if ds.pad1 > 0 else (ds.n1max + 1) * ds.sz]
        mask = mask.numpy().astype(np.uint8)

        # img = tiff.imread(os.path.join(DATA, idx + '.tiff'))
        # real_mask = row["encoding"]
        # if len(img.shape) == 5 or img.shape[0] == 3: 
        #     img = np.transpose(img.squeeze(), (1,2,0))
        # real_mask = enc2mask([real_mask], img.shape[:2][::-1])
        # x1 = mask * real_mask == 1
        # tps.append(x1.sum())
        # img[x1,2] = (img[x1,2] * 0.5 + 255 * 0.5).round().astype(np.uint8)
        # x1 = mask - real_mask == 1
        # fp = x1.sum()
        # fps.append(fp)
        # print("FP: ", fp)
        # img[x1,1] = (img[x1,1] * 0.5 + 255 * 0.5).round().astype(np.uint8)
        # x1 = real_mask - mask == 1
        # fn = x1.sum()
        # fns.append(fn)
        # print("FN: ", fn)
        # img[x1,0] = (img[x1,0] * 0.5 + 255 * 0.5).round().astype(np.uint8)
        # img = cv2.resize(img, (img.shape[1] // 15,img.shape[0] // 15), interpolation = cv2.INTER_AREA)
        # cv2.imwrite(f"result/{idx}_train.png", img)

        rle = rle_encode_less_memory(mask)
        names.append(idx)
        preds.append(rle)

        del mask, ds, dl, line, mp
        gc.collect()

    df = pd.DataFrame({'id': names, 'predicted': preds})
    df.loc[df.predicted == "", "predicted"] = "0 1"
    df.to_csv(f'./submission/submission_{suffix}_{model_idx}.csv', index=False)
    df.head()
    break
