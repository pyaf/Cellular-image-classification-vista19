import pdb
import os
import cv2
import time
from glob import glob
import torch
import scipy
import pandas as pd
import numpy as np
from PIL import Image
import jpeg4py as jpeg
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import albumentations
from albumentations import torch as AT
from torchvision.datasets.folder import pil_loader
import torch.utils.data as data
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import cohen_kappa_score, accuracy_score
from models import Model, get_model
from dataloader import *
from extras import *
from augmentations import *
from utils import *
from image_utils import *
from preprocessing import *


def test_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        dest="filepath",
        help="experiment config file",
        metavar="FILE",
        required=True,
    )
    parser.add_argument(
        "-e",
        "--epoch_range",
        nargs="+",
        type=int,
        dest="epoch_range",
        help="Epoch to start from",
    )  # usage: -e 10 20
    parser.add_argument(
        "-p",
        "--predict_on",
        dest="predict_on",
        help="predict on train or test set, options: test or train",
        default="test",
    )
    return parser


def get_predictions(model, testset):
    """return all predictions on testset in a list"""
    predictions = []
    for i, batch in enumerate(tqdm(testset)):
        _, images, _ = batch
        preds = model(images.to(device))
        preds = preds.detach().tolist()  # [1]
        predictions.extend(preds)
    return np.array(predictions)


if __name__ == "__main__":
    """
    Generates predictions on train/test set using the ckpts saved in the model folder path and saves them in npy_folder in npy format which can be analyses later for different thresholds
    """
    parser = test_parser()
    args = parser.parse_args()
    predict_on = args.predict_on
    start_epoch, end_epoch = args.epoch_range
    cfg = load_cfg(args)

    cfg['phase'] = "test" # "train" -> augmentations
    cfg['data_folder'] = 'data/test_final/'
    if predict_on == "train":
        cfg['sample_submission'] = "data/train.csv"
        cfg['data_folder'] = "data/train_final/"

    tta = 0  # number of augs in tta

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    test_dataloader = testprovider(cfg)

    model = get_model(cfg['model_name'], cfg['num_classes'], pretrained=None)
    model.to(device)
    model.eval()
    folder = os.path.splitext(os.path.basename(args.filepath))[0]
    model_folder_path = os.path.join('weights', folder)

    print(f"Saving predictions at: {model_folder_path}")
    print(f"From epoch {start_epoch} to {end_epoch}")
    print(f"Using tta: {tta}\n")
    df = pd.read_csv(cfg['sample_submission'])
    y_train = pd.read_csv(cfg['df_path'])['label'].values
    for epoch in range(start_epoch, end_epoch + 1):
        print(f"Using ckpt{epoch}.pth")
        ckpt_path = os.path.join(model_folder_path, "ckpt%d.pth" % epoch)
        state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(state["state_dict"])
        preds = get_predictions(model, test_dataloader)
        cls_preds = np.argmax(preds, axis=1).flatten() + 1 # +1 is fucking imp
        #print(np.unique(cls_preds, return_counts=True)[1])
        if predict_on == 'train':
            cm = ConfusionMatrix(y_train, cls_preds)
            acc = cm.overall_stat["Overall ACC"]
            print(f'ACC: {acc}\n')
        df.loc[:, "label"] = cls_preds
        path = os.path.join(model_folder_path, f"{predict_on}_ckpt{epoch}.csv")
        df.to_csv(path, index=False)
        print("Predictions saved!")


"""
Footnotes

[1] a cuda variable can be converted to python list with .detach() (i.e., grad no longer required) then .tolist(), apart from that a cuda variable can be converted to numpy variable only by copying the tensor to host memory by .cpu() and then .numpy
"""
