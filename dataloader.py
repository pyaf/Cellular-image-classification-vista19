import os
import cv2
import pdb
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, sampler
from torchvision.datasets.folder import pil_loader
from sklearn.model_selection import train_test_split, StratifiedKFold
from PIL import Image
import jpeg4py as jpeg
from extras import *
from image_utils import *
from augmentations import * #get_transforms
from preprocessing import *



class ImageDataset(Dataset):
    """training dataset."""

    def __init__(self, df, phase, cfg):
        """
        Args:
                fold: for k fold CV
                images_folder: the folder which contains the images
                df_path: data frame path, which contains image ids
                transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.phase = phase
        self.df = df
        self.size = cfg['size']
        self.num_samples = self.df.shape[0]
        self.fnames = self.df["id"].values
        self.labels = self.df["label"].values.astype("int64") - 1
        self.num_classes = cfg['num_classes']
        # self.labels = to_multi_label(self.labels, self.num_classes)  # [1]
        # self.labels = np.eye(self.num_classes)[self.labels]
        self.transform = get_transforms(phase, cfg)
        self.root = os.path.join(cfg['home'], cfg['data_folder'])

        '''
        self.images = []
        for fname in tqdm(self.fnames):
            path = os.path.join(self.images_folder, "bgcc300", fname + ".npy")
            image = np.load(path)
            self.images.append(image)
        '''

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        label = self.labels[idx]
        path = os.path.join(self.root, fname)
        # image = np.array(Image.open(path))
        image = cv2.imread(path)
        image = self.transform(image=image)["image"]
        return fname, image, label

    def __len__(self):
        #return 100
        return len(self.df)


def get_sampler(df, cfg):
    if cfg['cw_sampling']:
        '''sampler using class weights (cw)'''
        class_weights = cfg['class_weights']
        print("weights", class_weights)
        dataset_weights = [class_weights[idx] for idx in df["diagnosis"]]
        datasampler = sampler.WeightedRandomSampler(dataset_weights, len(df))
    if cfg['he_sampling']:
        '''sampler using hard examples (he)'''
        print('Hard example sampling')
        dataset_weights = df["weight"].values
        datasampler = sampler.WeightedRandomSampler(dataset_weights, len(df))
    else:
        datasampler = None
    return datasampler

def resampled(df, count_dict):
    ''' resample from df with replace=False'''
    def sample(obj):  # [5]
        return obj.sample(n=count_dict[obj.name], replace=False, random_state=69)
    sampled_df = df.groupby('diagnosis').apply(sample).reset_index(drop=True)
    return sampled_df


def provider(phase, cfg):
    HOME = cfg['home']
    df = pd.read_csv(os.path.join(HOME, cfg['df_path']))
    df['weight'] = 1 # [10]

    #print(df['diagnosis'].value_counts())
    fold = cfg['fold']
    total_folds = cfg['total_folds']
    kfold = StratifiedKFold(total_folds, shuffle=True, random_state=69)
    train_idx, val_idx = list(kfold.split(df["id"], df["label"]))[fold]
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

    if phase == "train":
        df = train_df.copy()
    elif phase == "val":
        df = val_df.copy()

    print(f"{phase}: {df.shape}")

    df = df.sample(frac=1, random_state=69) # shuffle


    image_dataset = ImageDataset(df, phase, cfg)

    datasampler = None
    if phase == "train":
        datasampler = get_sampler(df, cfg)
    print(f'datasampler: {datasampler}')

    batch_size = cfg['batch_size'][phase]
    num_workers = cfg['num_workers']
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False if datasampler else True,
        sampler=datasampler,
    )  # shuffle and sampler are mutually exclusive args

    #print(f'len(dataloader): {len(dataloader)}')
    return dataloader



def testprovider(cfg):
    HOME = cfg['home']
    df_path = cfg['sample_submission']
    df = pd.read_csv(os.path.join(HOME, df_path))
    phase = cfg['phase']
    if phase == "test":
        df['id_code'] += '.png'
    batch_size = cfg['batch_size']['test']
    num_workers = cfg['num_workers']


    dataloader = DataLoader(
        ImageDataset(df, phase, cfg),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False
    )
    return dataloader


if __name__ == "__main__":
    ''' doesn't work, gotta set seeds at function level
    seed = 69
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    '''
    import os
    os.environ['CUDA_VISIBLE_DEVICES']=""
    import torchvision
    torchvision.set_image_backend('accimage')

    import time
    start = time.time()
    phase = "train"
    args = get_parser()
    cfg = load_cfg(args)
    cfg["num_workers"] = 8
    cfg["batch_size"]["train"] = 4
    cfg["batch_size"]["val"] = 4

    dataloader = provider(phase, cfg)
    ''' train val set sanctity
    #pdb.set_trace()
    tdf = dataloader.dataset.df
    phase = "val"
    dataloader = provider(phase, cfg)
    vdf = dataloader.dataset.df
    print(len([x for x in tdf.id_code.tolist() if x in vdf.id_code.tolist()]))
    exit()
    '''
    total_labels = []
    total_len = len(dataloader)
    from collections import defaultdict
    fnames_dict = defaultdict(int)
    for idx, batch in enumerate(dataloader):
        fnames, images, labels = batch
        for fname in fnames:
            fnames_dict[fname] += 1

        print("%d/%d" % (idx, total_len), images.shape, labels.shape)
        total_labels.extend(labels.tolist())
        #pdb.set_trace()
    print(np.unique(total_labels, return_counts=True))
    diff = time.time() - start
    print('Time taken: %02d:%02d' % (diff//60, diff % 60))

    print(np.unique(list(fnames_dict.values()), return_counts=True))
    #pdb.set_trace()


"""
Footnotes:

https://github.com/btgraham/SparseConvNet/tree/kaggle_Diabetic_Retinopathy_competition

[1] CrossEntropyLoss doesn't expect inputs to be one-hot, but indices
[2] .value_counts() returns in descending order of counts (not sorted by class numbers :)
[3]: bad_indices are those which have conflicting diagnosises, duplicates are those which have same duplicates, we shouldn't let them split in train and val set, gotta maintain the sanctity of val set
[4]: used when the dataframe include external data and we want to sample limited number of those
[5]: as replace=False,  total samples can be a finite number so that those many number of classes exist in the dataset, and as the count_dist is approx, not normalized to 1, 7800 is optimum, totaling to ~8100 samples

[6]: albumentations.Normalize will divide by 255, subtract mean and divide by std. output dtype = float32. ToTensor converts to torch tensor and divides by 255 if input dtype is uint8.
[7]: indices of hard examples, evaluated using 0.81 scoring model.
[10]: messidor df append will throw err when doing hard ex sampling.
[11]: using current comp's data as val set in old data training.
[12]: messidor's class 3 is class 3 and class 4 combined.
"""
