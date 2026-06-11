# -*- coding: utf-8 -*-
"""Régénère la figure pca_approximatrion.png en 256x256 (au lieu de 64x64)."""
import os, sys
import matplotlib
matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..", "code")
FIGS = os.path.join(HERE, "..", "figures")
sys.path.insert(0, CODE)

import Script01_PreprocessingExploration as S

TS = (256, 256)

print("Loading dataset...")
bw_imgs, bw_dogs, labels, label_names = S.read_and_crop_db(color=False)
assert bw_dogs is not None, "SmallDB introuvable"

print(f"Resizing {len(bw_dogs)} images to {TS} (continuous padding)...")
resized = S.get_resized_db(bw_dogs, target_size=TS, pad_type='continuous')
data_mtx = S.convert_ndarrays2data_matrix(resized)

data_train, data_test, y_train, y_test = S.train_test_split(
    data_mtx, labels, test_size=0.25, stratify=labels, random_state=42)

print(f"Fitting PCA on {data_train.shape} ...")
pca = S.my_PCA(data_train)

out = os.path.join(FIGS, "pca_approximatrion.png")
S.display_pca_approx(data_mtx[0], pca, target_size=TS, fig_path=out)
print("OK ->", out)
