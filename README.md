# Intro-ML — Projet d'Introduction au Machine Learning

**Polytech Nice Sophia · MAM3 · 8–12 juin 2026**
Encadrants : Mahmoud Elsawy & Jean-Luc Bouchot (INRIA)

---

Ce dépôt contient le travail du groupe pour le projet d'initiation au Machine Learning. L'objectif principal est de construire un **pipeline de classification d'images de chiens** (dataset PASCAL VOC) en Python.

## Structure du dépôt

```
Intro-ML/
├── README.md
├── .gitignore
├── code/
│   └── Script01_PreprocessingExploration.py   # Script principal à compléter
├── SmallDB/                                    # Dataset (voir section setup)
│   ├── Images/                                 # Images par race de chien
│   └── Annotation/                             # Bounding boxes (fichiers XML Pascal VOC)
└── figures/                                    # Figures générées (créé automatiquement)
```

## Mise en place de l'environnement

```bash
# Cloner ce dépôt
git clone https://github.com/BnRomain/Intro-ML.git
cd Intro-ML

# Créer l'environnement virtuel
python -m venv MLPythonVenv

# Activer (Linux/Mac)
source MLPythonVenv/bin/activate
# Activer (Windows)
MLPythonVenv\Scripts\activate

# Installer les dépendances
pip install jupyter seaborn pandas scikit-learn scikit-image matplotlib numpy
```

> **Dataset** : cloner le repo de l'école (`git clone git@github.com:pns-mam/ml-intro.git`) puis copier le dossier `SmallDB/` à la racine de ce repo.

Pour lancer le script principal :
```bash
cd code
python Script01_PreprocessingExploration.py
```

---

## Pipeline général

Le script suit ce pipeline :

```
Images brutes
    -> Lecture + recadrage (bounding box XML)
    -> Redimensionnement uniforme 64x64
    -> Conversion en matrice de données
    -> PCA (réduction dimensionnelle + features)
    -> HOG (features de gradients)
    -> Combinaison PCA + HOG
    -> Normalisation (StandardScaler)
    -> Classification (kNN, SVM...)
    -> Evaluation (train/test error, confusion matrix)
```

---

## Tâches à implémenter

Le script contient 5 sections `### STUDENT IMPLEMENTATION ###` à compléter.

---

### Tâche 1 — Entropie de Shannon (`entropy`)

**Où :** fonction `entropy(p)`, section 2 Exploratory Data Analysis

**Objectif :** Mesurer l'équilibre des classes dans le dataset. Une entropie maximale indique des classes parfaitement équilibrées.

La formule de l'entropie de Shannon d'une distribution discrète P = (p_1, ..., p_n) :

```
H(P) = -sum_i [ p_i * ln(p_i) ]
```

Ce qu'il faut faire :
- Convertir `p` en tableau numpy float
- Normaliser pour que la somme vaille 1 (probabilités empiriques)
- Exclure les p_i = 0 avant d'appliquer le log (éviter `log(0) = -inf`)

---

### Tâche 2 — Redimensionnement avec préservation du ratio (`resize_and_pad`)

**Où :** fonction `resize_and_pad(img, target_size, pad_type)`, section 4

**Objectif :** Mettre toutes les images à une taille fixe (64x64) sans déformer les chiens. Stratégie : calculer le ratio de mise à l'échelle, redimensionner en gardant les proportions, puis centrer sur un canvas vide.

Ce qu'il faut calculer :
- `h_r = target_height / image_height` et `w_r = target_width / image_width`
- Le ratio à appliquer = `min(h_r, w_r)` pour ne pas déborder
- `new_h = int(h * ratio)` et `new_w = int(w * ratio)` — nouvelles dimensions
- Les coordonnées `bottom, top, left, right` pour centrer l'image redimensionnée sur le canvas

Trois types de padding requis :
- `'white'` : canvas initialisé à 1.0 (déjà géré après votre code)
- `'black'` : canvas initialisé à 0.0 (déjà géré après votre code)
- `'continuous'` : remplissage avec la valeur du pixel le plus proche du bord

---

### Tâche 3.1 — Projection PCA (`project_onto_PCA`)

**Où :** fonction `project_onto_PCA(n_components, pca_model, data)`, section 5

**Objectif :** Projeter les données sur les `n_components` premières composantes principales d'un modèle PCA déjà entraîné.

Ce qu'il faut faire :
- Utiliser `pca_model.transform(data)` pour projeter
- Retourner uniquement les `n_components` premières colonnes du résultat

---

### Tâche 3.2 — Scree Plot PCA (`visualize_var_pcs`)

**Où :** fonction `visualize_var_pcs(pca, fig_path)`, section 5

**Objectif :** Visualiser quelle proportion de la variance est capturée par chaque composante principale.

Ce qu'il faut tracer :
- Un barplot de `pca.explained_variance_ratio_` (variance individuelle par composante)
- Une courbe de `np.cumsum(explained_variance_ratio_)` (variance cumulée)
- Une ligne horizontale de référence à 90% pour repérer le nombre minimal de composantes à conserver
- Retourner la figure dans la variable `fig`

---

### Tâche 3.3 — Scatter plot 2D PCA (`plot_whole_db_on_2d`)

**Où :** fonction `plot_whole_db_on_2d(pca, data_mtx, fig_path)`, section 5

**Objectif :** Projeter tout le dataset sur les 2 premières composantes principales et vérifier visuellement si les classes sont séparables.

Ce qu'il faut faire :
- Appeler `project_onto_PCA` pour obtenir les coordonnées 2D de chaque image
- Créer un scatter plot coloré par classe (une couleur par race de chien)
- Ajouter une légende avec les noms de races
- Retourner la figure dans la variable `fig`

Bonus (non obligatoire) : tenter un scatter plot 3D avec les 3 premières composantes.

---

### Tâche 4 — Reconstruction PCA (`display_pca_approx`)

**Où :** fonction `display_pca_approx(img, pca, target_size, fig_path)`, section 5

**Objectif :** Montrer qu'avec suffisamment de composantes principales on peut reconstruire une image fidèlement. Visualiser l'évolution de la qualité de reconstruction en fonction du nombre de composantes k.

La reconstruction à partir de k composantes principales :

```
x_hat_k = mean + sum_{i=1}^{k} <x - mean, u_i> * u_i
```

En pratique avec scikit-learn : projeter sur les k premiers axes, mettre à zéro les composantes > k, puis appeler `pca.inverse_transform(...)`.

Ce qu'il faut faire :
- Boucler sur plusieurs valeurs de k (ex. 1, 5, 10, 20, 50, 100, ...)
- Pour chaque k : reconstruire l'image et calculer l'erreur L2 `||original - reconstruction||`
- Afficher chaque reconstruction dans un subplot (la grille 4x4 est déjà initialisée)

---

### Tâche 5 — HOG : Histogramme de Gradients Orientés (`compute_hog`)

**Où :** fonction `compute_hog(image, nb_height_cells, nb_width_cells, nb_bins)`, section 5

**Objectif :** Extraire des features classiques de texture basées sur les orientations locales des contours. Ces features complètent les features PCA pour la classification.

Principe algorithmique :
1. Les gradients `g_x` (Sobel horizontal) et `g_y` (Sobel vertical) sont déjà calculés
2. Magnitude : `|g| = sqrt(g_x^2 + g_y^2)`, Orientation : `theta = arctan2(g_y, g_x)` dans `[0, pi)`
3. Diviser l'image en une grille de `nb_height_cells x nb_width_cells` cellules
4. Dans chaque cellule, accumuler les magnitudes dans `nb_bins` intervalles angulaires équirépartis sur `[0, pi)`

Ce qu'il faut implémenter :
- Boucler sur chaque cellule `(i, j)` dans la grille
- Extraire la région correspondante de `magnitude` et `orientation`
- Pour chaque pixel : trouver le bon bin avec `int(theta / bin_width) % nb_bins` et y ajouter la magnitude
- Stocker dans `output[i, j, bin]`

---

### Analyse : Impact du StandardScaler

Une fois le pipeline complet opérationnel, faire l'expérience suivante :

1. Lancer le script tel quel (avec `StandardScaler`) — noter train error et test error
2. Commenter les 3 lignes du `StandardScaler` et passer `X_train_combined` / `X_test_combined` directement au kNN
3. Observer et **expliquer dans le rapport** pourquoi la différence est aussi marquée

---

## Livrables

| Livrable | Détail |
|----------|--------|
| Script complété | `Script01_PreprocessingExploration.py` avec toutes les tâches |
| Figures | Dossier `figures/` avec les plots générés |
| Rapport | 5 pages max, PDF, analyse des résultats et choix effectués |
| Présentation | Courte présentation orale |

**Deadline : jeudi 11 juin 2026, 17h00**
Envoyer à `mahmoud.elsawy@inria.fr` et `jean-luc.bouchot@inria.fr`

---

## Calendrier

| Jour | Contenu |
|------|---------|
| Lundi 8 juin | Setup, chargement des données, Tâches 1 et 2 |
| Mardi 9 juin | Tâches 3.1/3.2/3.3 (PCA), Tâche 4 (reconstruction) |
| Mercredi 10 juin | Tâche 5 (HOG), SVM, hyperparamètres, comparaison d'algorithmes |
| Jeudi 11 juin matin | Finalisation rapport + envoi avant 17h |
| Vendredi 12 juin matin | Evaluation |

---

## Ressources utiles

- [Repo de l'école](https://github.com/pns-mam/ml-intro)
- [Documentation scikit-learn](https://scikit-learn.org/stable/)
- [PCA scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [Documentation skimage](https://scikit-image.org/docs/stable/)
