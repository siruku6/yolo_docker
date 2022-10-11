import os
import shutil
from glob import glob
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

PARENT_DIR: str = "../VOCdevkit/VOC2008"


def extract_filenames(target_file_name: str, ext: str = '.jpg') -> List[str]:
    target_dir: str = "ImageSets/Main"
    dir_name: str = os.path.join(PARENT_DIR, target_dir)

    df = pd.read_csv(os.path.join(dir_name, target_file_name), header=None, sep=" ")

    filename_ndarray: np.ndarray = df[0].to_numpy()
    filenames: List[str] = [
        filename + ext for filename in filename_ndarray
    ]
    return filenames


def copy_files(fname_list: List[str], origin_dir: str, target_dir: str):
    os.makedirs(target_dir, exist_ok=True)

    for fname in tqdm(fname_list):
        file_path: str = os.path.join(origin_dir, fname)
        shutil.copy(file_path, target_dir)


##################################################################
# Move image files
##################################################################
train_filenames: List[str] = extract_filenames(target_file_name="train_train.txt")
val_filenames: List[str] = extract_filenames(target_file_name="train_val.txt")

origin_dir: str = os.path.join(PARENT_DIR, "JPEGImages")
copy_files(train_filenames, origin_dir, target_dir="../tr_data/images/train")
copy_files(val_filenames, origin_dir, target_dir="../tr_data/images/val")


##################################################################
# Move annotation files
##################################################################
train_annotation_filenames: List[str] = extract_filenames(target_file_name="train_train.txt", ext='.txt')
val_annotation_filenames: List[str] = extract_filenames(target_file_name="train_val.txt", ext='.txt')

origin_dir: str = os.path.join("../tr_data", "format4yolo")
copy_files(train_annotation_filenames, origin_dir, target_dir="../tr_data/labels/train")
copy_files(val_annotation_filenames, origin_dir, target_dir="../tr_data/labels/val")
