# -*- coding: utf-8 -*-
"""Appelle plot_barplot du Script01 avec les vrais labels de SmallDB et sauvegarde la figure."""
import os, sys
import matplotlib
matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..", "code")
FIGS = os.path.join(HERE, "..", "figures")
sys.path.insert(0, CODE)

import Script01_PreprocessingExploration as S

# Vrais comptes lus dans SmallDB/Images (ordre des dossiers triés)
breeds = [
    ("Chihuahua", 152),
    ("basset", 175),
    ("Kerry_blue_terrier", 179),
    ("groenendael", 150),
    ("malinois", 150),
    ("chow", 196),
]
label_names = {i: name for i, (name, _) in enumerate(breeds)}
labels = []
for i, (_, n) in enumerate(breeds):
    labels += [i] * n

out = os.path.join(FIGS, "class_balance.png")
S.plot_barplot(labels, label_names, fig_path=out)
print("OK ->", out)
