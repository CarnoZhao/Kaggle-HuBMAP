sz = 512  # the size of tiles
reduce = 1024 // sz  # reduce the original images by 4 times
TH = 0.4  # threshold for positive predictions
bs = 64
DATA = './data/raw/test'
csv = 'data/raw/sample_submission.csv'
suffix = "051408"
suffix2 = ""
import os
MODELS = [
    *os.popen(f'find ./logs/ -name "*epoch*.ckpt" | grep {suffix}').read().split(),
]
name = "timm-efficientnet-b2"
attention = False

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
# from utils.seg.UneXt import UneXt50
import pytorch_lightning as pl

# df_sample = pd.read_csv('data/raw/sample_submission.csv')
# df_sample = pd.read_csv('data/raw/train.csv')
df_sample = pd.read_csv(csv)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mean = np.array([0.63701425, 0.47097038, 0.68173952])
std = np.array([0.15979014, 0.22442915, 0.14194921])

s_th = 40  # saturation blancking threshold
p_th = 1000*(sz//256)**2 # threshold for the minimum number of pixels
identity = rasterio.Affine(1, 0, 0, 0, 1, 0)

def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))

def rle_encode_less_memory(img):
    # the image should be transposed
    pixels = img.T.flatten()

    # This simplified method requires first and last pixel to be zero
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)


class HuBMAPDataset(Dataset):
    def __init__(self, idx, sz=sz, reduce=reduce, shift = (0, 0)):
        self.data = rasterio.open(os.path.join(DATA,idx+'.tiff'), transform = identity,num_threads='all_cpus')
        if self.data.count != 3:
            self.layers = [rasterio.open(subd) for subd in self.data.subdatasets]
        self.shift = shift
        self.shape = list(self.data.shape)
        self.shape[0] -= self.shift[0]
        self.shape[1] -= self.shift[1]
        self.reduce = reduce
        self.sz = reduce*sz
        self.pad0 = (self.sz - self.shape[0] % self.sz) % self.sz
        self.pad1 = (self.sz - self.shape[1] % self.sz) % self.sz
        self.n0max = (self.shape[0] + self.pad0)//self.sz
        self.n1max = (self.shape[1] + self.pad1)//self.sz

    def __len__(self):
        return self.n0max*self.n1max

    def __getitem__(self, idx):
        n0, n1 = idx//self.n1max, idx % self.n1max
        x0, y0 = -self.pad0//2 + n0*self.sz, -self.pad1//2 + n1*self.sz
        # make sure that the region to read is within the image
        p00, p01 = max(0, x0), min(x0+self.sz, self.shape[0])
        p10, p11 = max(0, y0), min(y0+self.sz, self.shape[1])
        img = np.zeros((self.sz, self.sz, 3), np.uint8)
        # mapping the loade region to the tile

        if self.data.count == 3: # normal
            image = self.data.read([1,2,3], window=Window.from_slices((p00+self.shift[0],p01+self.shift[0]),(p10+self.shift[1],p11+self.shift[1])))
            image = np.moveaxis(image, 0, -1)
        else: # with subdatasets/layers
            image = np.zeros((p01-p00, p11-p10, 3), dtype=np.uint8)
            for fl in range(3):
                image[:,:,fl] = self.layers[fl].read(window=Window.from_slices((p00+self.shift[0],p01+self.shift[0]),(p10+self.shift[1],p11+self.shift[1])))

        img[(p00-x0):(p01-x0), (p10-y0):(p11-y0)] = image
        # (36800, 43780)
        if self.reduce != 1:
            img = cv2.resize(img, (self.sz//reduce, self.sz//reduce),
                             interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if (s > s_th).sum() <= p_th or img.sum() <= p_th:
            # images with -1 will be skipped
            return img2tensor((img/255.0 - mean)/std), -1
        else:
            return img2tensor((img/255.0 - mean)/std), idx


class Model_pred:
    def __init__(self, models, dl, tta: bool = False, half: bool = False):
        self.models = models
        self.dl = dl
        self.tta = tta
        self.half = half

    def __iter__(self):
        count = 0
        with torch.no_grad():
            for x, y in tqdm(iter(self.dl), total = len(self.dl)):
                if ((y >= 0).sum() > 0):  # exclude empty images
                    x = x[y >= 0].to(device)
                    y = y[y >= 0]
                    if self.half:
                        x = x.half()
                    py = None
                    for model in self.models:
                        p = model(x)
                        p = torch.sigmoid(p).detach()
                        if py is None:
                            py = p
                        else:
                            py += p
                    if self.tta:
                        # x,y,xy flips as TTA
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

                    # py = F.upsample(py, scale_factor=reduce, mode="bilinear")
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
        self.model = UnetPlusPlus(name, decoder_attention_type = None, encoder_weights = None)
        self.criterion = None #Criterion(loss_weight = np.array([1., 3.]))

    def forward(self, x):
        return self.model(x)


models = []
state_dict = None
for path in MODELS:
    # state_dict = torch.load(path, map_location=torch.device('cuda'))
    model = Model()# UneXt50()#Unet(name, encoder_weights=None, decoder_attention_type="scse" if attention else None)
    model = model.load_from_checkpoint(path)# load_state_dict(state_dict)
    model.float()
    model.eval()
    model.to(device)
    models.append(model)

del state_dict

names, preds = [], []

for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
    idx = row['id']
    # if not idx.startswith("d48"):
    #     continue
    # rasterio cannot be used with multiple workers
    shifts = [(0, 0), (0, sz), (sz, 0), (sz, sz)]
    
    MASK = None
    for shift in shifts:
        ds = HuBMAPDataset(idx, shift = shift)
        dl = DataLoader(ds, bs, num_workers=0, shuffle=False, pin_memory=True)
        mp = Model_pred(models, dl)
        # generate masks
        mask = torch.zeros(len(ds), ds.sz // reduce, ds.sz // reduce, dtype=torch.float32)
        for p, i in iter(mp):
            mask[i.item()] += p.squeeze(-1)# > TH

        # reshape tiled masks into a single mask and crop padding
        mask = mask.view(ds.n0max, ds.n1max, ds.sz // reduce, ds.sz // reduce).\
            permute(0, 2, 1, 3).reshape(ds.n0max*(ds.sz // reduce), ds.n1max*(ds.sz // reduce))
        mask = mask[ds.pad0//2//reduce:-(ds.pad0-ds.pad0//2)//reduce if ds.pad0 > 0 else ds.n0max*(ds.sz // reduce), ds.pad1//2//reduce:-(ds.pad1-ds.pad1//2)//reduce if ds.pad1 > 0 else ds.n1max*(ds.sz // reduce)]
        if shift == (0, 0):
            target_size = ds.shape
        # MASK[shift[0]:,shift[1]:] += mask
        if MASK is None:
            MASK = mask
        else:
            MASK[shift[0] // reduce:,shift[1] // reduce:] += mask
        del ds, dl, mask
        gc.collect()
    MASK[sz//2:,sz//2:] /= 4
    MASK[sz//2:,:sz//2] /= 2
    MASK[:sz//2,sz//2:] /= 2
    MASK = F.upsample(MASK[None,None], size = target_size[:2], mode="bilinear").squeeze()
    MASK = (MASK > TH).type(torch.int8)

    img = tiff.imread(os.path.join(DATA, idx+'.tiff'))
    if len(img.shape) == 5 or img.shape[0] == 3: 
        img = np.transpose(img.squeeze(), (1,2,0))
    img[MASK == 1,1] = img[MASK == 1,1] * 0.5 + 255 * 0.5
    img = img.astype(np.uint8)
    img = cv2.resize(img,(img.shape[1]//(reduce * 2),img.shape[0]//(reduce*2)),interpolation = cv2.INTER_AREA)
    cv2.imwrite(f"result/{idx}.png", img)
    print(MASK.sum())
    del img
    gc.collect()

    rle = rle_encode_less_memory(MASK)
    names.append(idx)
    preds.append(rle)
    del MASK
    # del img
    gc.collect()

df = pd.DataFrame({'id': names, 'predicted': preds})
df.loc[df.predicted == "", "predicted"] = "0 1"
df.to_csv(f'./submission/submission_{suffix}.csv', index=False)
df.head()
