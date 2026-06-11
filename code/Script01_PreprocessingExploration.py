#### Import necessary libraries
# General libraries
import os
import random  # Random number generators
import xml.etree.ElementTree as ET  # For parsing XML like documents
# Numerical libraries
import numpy as np

# Visualisation
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Learning
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, hamming_loss
from sklearn.neighbors import KNeighborsClassifier

# Image processing
import skimage.io as io  # image I/O routines
from skimage.transform import rescale, resize
from skimage.filters import sobel, sobel_h, sobel_v, gaussian

# We will add some macros to facilitate work later on. This is often good practice...
# Adapt here if you changed your workspace
##########################################
## Useful Macros & Parameters
##########################################
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, '..')

PATH_TO_DB = os.path.join(PROJECT_DIR, 'SmallDB')
IMG_DB = 'Images'
ANNOT_DB = 'Annotation'
IMG_FULL = os.path.join(PATH_TO_DB, IMG_DB)

TARGET_SIZE = (64, 64)

FIGS = os.path.join(PROJECT_DIR, 'figures')
os.makedirs(FIGS, exist_ok=True)

DATA_DIR = PROJECT_DIR
os.makedirs(DATA_DIR, exist_ok=True)


##########################################
## 1. Load Dataset
##########################################

def read_img(n_value, dog_breed, fname, TO_DB=IMG_FULL, color=True):
    """Reads a single image using skimage.io."""
    folder_name = f"{n_value}-{dog_breed}"
    img_path = os.path.join(TO_DB, folder_name, fname)
    return io.imread(img_path, as_gray=not color)


def read_db(subset_dogs=None, TO_DB=IMG_FULL, color=True):
    """Loads images, numeric labels, and human-readable label mapping."""
    label_names = dict()
    next_label = 0
    labels = []
    images = []

    if not os.path.exists(TO_DB):
        return None, None, None

    for subdir in sorted(os.listdir(TO_DB)):
        if '-' not in subdir:
            continue
        n_value, dog_breed = subdir.split('-', 1)

        if subset_dogs and dog_breed not in subset_dogs:
            continue

        label_names[next_label] = dog_breed
        nb_img = 0

        for img in os.listdir(os.path.join(TO_DB, subdir)):
            images.append(read_img(n_value, dog_breed, img, color=color))
            nb_img += 1

        labels.extend([next_label] * nb_img)
        next_label += 1

    return images, labels, label_names


##########################################
## 2. Exploratory Data Analysis
##########################################

def entropy(p):
    """
    ========================================================================
    TASK 1: SHANNON ENTROPY
    ========================================================================
    Computes the Shannon entropy of a discrete distribution to evaluate
    class balance in the dataset. Maximum entropy indicates perfectly
    balanced classes.

    Formula: H(P) = -sum_i [ p_i * ln(p_i) ]

    Steps:
    - Convert p to a numpy float array
    - Normalize so probabilities sum to 1 (empirical probabilities)
    - Exclude p_i = 0 before applying log (avoid log(0) = -inf)
    """
    entropy_value = 0.0
    p = np.array(p)
    p = p / np.sum(p)
    p = p[p != 0]
    entropy_value = -np.sum(p * np.log(p))
    return entropy_value


def entropy_breeds(labels):
    """Computes distribution entropy for evaluation of class balance."""
    _, counts = np.unique(labels, return_counts=True)
    return entropy(counts)


def plot_barplot(labels, label_names, fig_path=None):
    """Generates class balance bar plots."""
    l, v = np.unique(labels, return_counts=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(l, v, color='#2e75b6', edgecolor='black', alpha=0.8)
    ax.set_xticks(l)
    ax.set_xticklabels([label_names[_l] for _l in l], rotation=15, ha='right')
    ax.set_ylabel("Sample Count")
    ax.set_title("SmallDB Class Balance Evaluation", fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    if fig_path:
        plt.savefig(fig_path, bbox_inches='tight', dpi=150)
    plt.close()
    return fig


##########################################
## 3. Localization and Preprocessing
##########################################

def plot_images(imgs, labels, label_names, nb_rows=3, nb_cols=4, fig_path=None, rdm_indices=None):
    if not rdm_indices:
        # Random selection of indices from the database
        rdm_indices = random.sample(range(0, len(imgs)), nb_rows * nb_cols)
    fig, ax = plt.subplots(nb_rows, nb_cols, figsize=(10, 5))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9,
                        top=0.9, wspace=0.4, hspace=0.4)
    for i in range(nb_rows):
        for j in range(nb_cols):
            ax[i][j].imshow(imgs[rdm_indices[i * nb_rows + j]], cmap='gray')
            ax[i][j].set_title(label_names[labels[rdm_indices[i * nb_rows + j]]])
    if fig_path:
        plt.savefig(fig_path, bbox_inches='tight', dpi=150)
        plt.show()
    plt.close()

    return rdm_indices, fig


def extract_bounding_box(fname):
    """Parses Pascal VOC XML files to retrieve object bounding coordinates."""
    tree = ET.parse(fname)
    root = tree.getroot()

    xmin = int(root.find('object/bndbox/xmin').text)
    ymin = int(root.find('object/bndbox/ymin').text)
    xmax = int(root.find('object/bndbox/xmax').text)
    ymax = int(root.find('object/bndbox/ymax').text)

    return xmin, xmax, ymin, ymax


def read_and_crop(n_value, dog_breed, fname, TO_DB=PATH_TO_DB, color=True):
    """Reads source image alongside its XML bounding box data to crop target."""
    img = io.imread(os.path.join(TO_DB, IMG_DB, f"{n_value}-{dog_breed}", fname), as_gray=not color)
    base_name = os.path.splitext(fname)[0]
    bndbox_name = os.path.join(TO_DB, ANNOT_DB, f"{n_value}-{dog_breed}", base_name)

    xmin, xmax, ymin, ymax = extract_bounding_box(bndbox_name)
    dog = img[ymin:ymax + 1, xmin:xmax + 1]

    return img, dog


def read_and_crop_db(subset_dogs=None, TO_DB=PATH_TO_DB, color=True):
    """Processes entire dataset, returning original and isolated target sub-arrays."""
    label_names = dict()
    next_label = 0
    labels = []
    images = []
    dogs = []

    img_root = os.path.join(TO_DB, IMG_DB)
    if not os.path.exists(img_root):
        return None, None, None, None

    for subdir in sorted(os.listdir(img_root)):
        if '-' not in subdir:
            continue
        n_value, dog_breed = subdir.split('-', 1)

        if subset_dogs and dog_breed not in subset_dogs:
            continue

        label_names[next_label] = dog_breed
        nb_img = 0

        for img_file in os.listdir(os.path.join(img_root, subdir)):
            img, dog = read_and_crop(n_value, dog_breed, img_file, TO_DB=TO_DB, color=color)
            images.append(img)
            dogs.append(dog)
            nb_img += 1

        labels.extend([next_label] * nb_img)
        next_label += 1

    return images, dogs, labels, label_names


##########################################
## 4. Dimensionality Reduction & Resizing
##########################################

def img_rescale(img, ratio):
    if len(img.shape) == 3:
        return rescale(img, ratio, anti_aliasing=True, channel_axis=2)
    return rescale(img, ratio, anti_aliasing=True)


def img_resize(img, target_size):
    return resize(img, target_size, anti_aliasing=True)


def resize_and_pad(img, target_size=TARGET_SIZE, pad_type='white'):
    """
    ========================================================================
    TASK 2: ASPECT-PRESERVING SPATIAL SCALING
    ========================================================================
    Resizes an image preserving its original aspect ratio, centering it
    on a fixed bounding canvas. At least three padding should be implemented
    * pad_type: Type of padding to use ('white' or 'black' or 'continuous').
        - 'white': Pads with maximum intensity (1.0 for normalized images).
        - 'black': Pads with minimum intensity (0.0 for normalized images).
        - 'continuous': Pads with the closest pixel value in the original image.
        - 'random': Pads with random pixels from the image.
        - 'propagated_blur': Pads with blurred propagation of edge pixels.
    Other potential padding strategies (e.g., reflection, edge replication, mean pixel values...) can be implemented as extensions.
    """

    h, w = img.shape[:2]
    multi_channel = len(img.shape) == 3
    height, width = target_size

    ### STUDENT IMPLEMENTATION START ###

    # 1. Compute scaling ratios for height and width
    h_r = height / h
    w_r = width / w
    if h_r < w_r:
        new_h = height
        new_w = int(w * h_r)
        top, bottom = 0, height
        left, right = (width - new_w) // 2, (width - new_w) // 2 + new_w
    else:
        new_h = int(h * w_r)
        new_w = width
        top, bottom = (height - new_h) // 2, (height - new_h) // 2 + new_h
        left, right = 0, width

    ### STUDENT IMPLEMENTATION END ###

    resized_img = img_resize(img, (new_h, new_w))

    output_shape = (height, width) + ((img.shape[2],) if multi_channel else ())
    output_img = np.ones(output_shape) if pad_type.lower() == 'white' else np.zeros(output_shape)

    if multi_channel:
        output_img[top:bottom, left:right, :] = resized_img
    else:
        output_img[top:bottom, left:right] = resized_img


    if pad_type.lower() == 'continuous':

        pad_top = top
        pad_bottom = height - bottom
        pad_left = left
        pad_right = width - right

        paddings = ((pad_top, pad_bottom), (pad_left, pad_right))
        if multi_channel:
            paddings += ((0, 0),)

        output_img = np.pad(resized_img, paddings, mode='edge')

    return output_img


def get_resized_db(img_in, target_size=TARGET_SIZE, pad_type='white'):
    if len(img_in) == 0: return np.array([])
    has_channels = len(img_in[0].shape) >= 3
    shape_tuple = (len(img_in), target_size[0], target_size[1]) + ((img_in[0].shape[2],) if has_channels else ())
    output = np.zeros(shape_tuple)
    for idx, img in enumerate(img_in):
        output[idx] = resize_and_pad(img, target_size=target_size, pad_type=pad_type)
    return output


def convert_ndarrays2data_matrix(arr):
    nb_individuals = len(arr)
    dim = np.prod(arr.shape[1:])
    data_mtx = np.zeros((nb_individuals, dim))
    for i in range(nb_individuals):
        data_mtx[i, :] = np.ravel(arr[i])
    return data_mtx


##########################################
## 5. Feature Engineering & Classification Pipeline
##########################################

def my_PCA(data, n_components=None):
    pca = PCA(n_components=n_components)
    pca.fit(data)
    return pca


def project_onto_PCA(n_components, pca_model, data):
    """
    ========================================================================
    TASK 3.1: CLASS-WISE PCA FEATURE EXTRACTION
    ========================================================================
    For a given PCA model, projects input data and extracts the top 'nb_components' principal components as features.
    """
    ### STUDENT IMPLEMENTATION START ###
    data = pca_model.transform(data)
    return data[:, 0:n_components]


def visualize_var_pcs(pca, fig_path=None):
    """
    ========================================================================
    TASK 3.2: PCA SCREE PLOT VISUALIZATION
    ========================================================================
    Plots the individual and cumulative explained variance ratios for each principal component, with a reference line at 90% cumulative variance.
    """
    exp_var_pca = pca.explained_variance_ratio_
    cum_var = np.cumsum(exp_var_pca)

    fig, ax1 = plt.subplots(figsize=(8, 4))

    # Barres : variance individuelle par composante
    ax1.bar(range(len(exp_var_pca)), exp_var_pca, color='#2e75b6', alpha=0.8)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Individual Explained Variance", color='#2e75b6')

    # Deuxième axe Y pour la courbe cumulée
    ax2 = ax1.twinx()
    ax2.plot(range(len(cum_var)), cum_var, color='red', marker='o', markersize=3)
    ax2.axhline(y=0.9, color='gray', linestyle='--', label='90% threshold')
    ax2.set_ylabel("Cumulative Explained Variance", color='red')
    ax2.legend(loc='center right')

    ax1.set_title("PCA Scree Plot", fontweight='bold')
    plt.tight_layout()

    if fig_path:
        plt.savefig(fig_path, bbox_inches='tight', dpi=150)
    plt.close()
    return fig


def plot_whole_db_on_2d(pca, data_mtx, fig_path=None):
    """
    ========================================================================
    TASK 3.3: PCA PROJECTION SCATTER PLOT
    ========================================================================
    Projects the entire dataset onto the first two principal components and visualizes it as a scatter plot, colored by class labels.
    """


    projected_data = project_onto_PCA(2, pca, data_mtx)

    fig, ax = plt.subplots(figsize=(8, 6))  # Légèrement agrandi pour la colorbar

    # 1. Déterminer dynamiquement le nombre de classes uniques
    unique_labels = np.unique(labels)
    nb_classes = len(unique_labels)

    # 2. On échantillonne exactement le nombre de couleurs nécessaires (ex: 6 au lieu de 10)
    cmap = plt.get_cmap('tab10', nb_classes)

    # 3. On crée des frontières discrètes pour centrer les tics (-0.5, 0.5, 1.5...)
    boundaries = np.arange(nb_classes + 1) - 0.5
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)

    # 4. On passe le 'cmap' et le 'norm' au scatter plot
    scatter = ax.scatter(
        projected_data[:, 0],
        projected_data[:, 1],
        c=labels,
        cmap=cmap,
        norm=norm,
        alpha=0.7,
        edgecolors='none'
    )

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("PCA 2D Projection", fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)

    # 5. On configure la colorbar pour afficher un tic propre au centre de chaque bloc
    cbar = fig.colorbar(scatter, ax=ax, ticks=unique_labels)
    cbar.set_label('Class')

    plt.tight_layout()

    if fig_path:
        plt.savefig(fig_path, bbox_inches='tight', dpi=150)
        plt.close()

    return fig


def display_pca_approx(img, pca, target_size=TARGET_SIZE, fig_path=None):
    """
    ========================================================================
    TASK 4: PCA RECONSTRUCTION APPROXIMATION
    ========================================================================
    For a given input image, reconstructs approximations using increasing numbers of principal components (k) and visualizes the original and reconstructed images side by side, along with their respective L2 reconstruction errors.
    """
    display_image = img.reshape(target_size[0], target_size[1])

    fig = plt.figure(figsize=(15, 15))
    plt.subplot(4, 4, 1)
    plt.imshow(display_image, cmap='gray')
    plt.title("Original image")
    plt.axis('off')

    step = (target_size[0] + target_size[1]) // 2

    k_values = [10, 110, 210, 310, 410, 510, 610, 710, 810, 910, 1010, 1110, 1210, 1310, 1410]

    for idx, k in enumerate(k_values[:15]):

        img_flat = img.reshape(1, -1)
        pca_proj = pca.transform(img_flat)[:, :k]

        reconstructed = np.dot(pca_proj, pca.components_[:k, :]) + pca.mean_
        recon_img = reconstructed.reshape(target_size)
        l2_err = np.linalg.norm(img_flat - reconstructed)

        plt.subplot(4, 4, idx + 2)
        plt.imshow(recon_img, cmap='gray')
        plt.title(f"k={k}, l2 err={l2_err:.2f}", fontsize=10)
        plt.axis('off')

    if fig_path:
        plt.savefig(fig_path, bbox_inches='tight', dpi=150)
    plt.close()
    return fig


def compute_pca_features(pcas, input_data, nb_components=5):
    out = np.zeros((len(input_data), 0))
    for pca in pcas:
        proj = pca.transform(input_data)
        features = proj[:, :min(proj.shape[1], nb_components)]
        out = np.hstack((out, features))
    return out


def compute_hog(image, nb_height_cells=4, nb_width_cells=4, nb_bins=8):
    """
    ========================================================================
    TASK 5: CLASSICAL HISTOGRAM OF ORIENTED GRADIENTS (HOG)
    ========================================================================
    Decomposes an image into local grid patches, evaluates edge orientations,
    and accumulates magnitudes within discrete orientation channels.
    """
    h_edge = sobel_h(image)
    v_edge = sobel_v(image)
    magnitude = np.sqrt(h_edge ** 2 + v_edge ** 2)

    orientation = np.arctan2(v_edge, h_edge)
    orientation[orientation < 0] += np.pi  # Constrain range within [0, \pi)

    H, W = image.shape[:2]
    cell_h = H // nb_height_cells
    cell_w = W // nb_width_cells
    bin_width = np.pi / nb_bins

    output = np.zeros((nb_height_cells, nb_width_cells, nb_bins))

    ### STUDENT IMPLEMENTATION START ###

    for i in range(nb_height_cells):
        for j in range(nb_width_cells):
            cell_magnitude = magnitude[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
            cell_orientation = orientation[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]

            for k in range(cell_h):
                for l in range(cell_w):
                    ang = cell_orientation[k, l]
                    mag = cell_magnitude[k, l]

                    index_bin = int(ang // bin_width)

                    if index_bin == nb_bins:
                        index_bin = nb_bins - 1

                    output[i, j, index_bin] += mag

    return output.reshape(-1)


# =========================================================================
if __name__ == '__main__':
    # Options: 'white', 'black', 'continuous', 'reflect', 'random', 'propagated_blur'
    CHOSEN_PAD_TYPE = 'continuous'

    print("Step 1: Loading Dataset")

    bw_imgs, bw_dogs, labels, label_names = read_and_crop_db(color=False)

    if bw_imgs is None:
        print(" BigDB not found.")
        label_names = {0: 'Chihuahua', 1: 'Pug', 2: 'Malamute', 3: 'Beagle'}
        labels = [0] * 152 + [1] * 250 + [2] * 300 + [3] * 300

        np.random.seed(42)
        sample_img = np.zeros((70, 70))
        sample_img[20:55, 15:55] = 0.6
        bw_imgs = [sample_img] * len(labels)
        bw_dogs = [sample_img[10:60, 10:60]] * len(labels)

        data_mtx = np.random.rand(len(labels), TARGET_SIZE[0] * TARGET_SIZE[1]) * 0.4
        for idx, lbl in enumerate(labels):
            data_mtx[idx, 400:1500] += lbl * 0.1
    else:
        print(" Real dataset successfully resolved.")
        resized_dogs = get_resized_db(bw_dogs, target_size=TARGET_SIZE, pad_type=CHOSEN_PAD_TYPE)
        data_mtx = convert_ndarrays2data_matrix(resized_dogs)

    # Calculate Shannon Entropy
    db_entropy = entropy_breeds(labels)
    print(f"   - Dataset Balance Shannon Entropy: {db_entropy:.4f}")
    plot_barplot(labels, label_names)
    plt.close()

    print("\n Step 2: Saving Localization & Bounding Box Crops")
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6, 3))
    ax0.imshow(bw_imgs[0], cmap='gray')
    ax0.set_title("Original Asset Image Frame", fontsize=9)
    ax0.axis('off')
    ax1.imshow(bw_dogs[0], cmap='gray')
    ax1.set_title("XML Isolated Target Region", fontsize=9)
    ax1.axis('off')
    plt.savefig(os.path.join(FIGS, 'bounding_box_crop.png'), bbox_inches='tight', dpi=150)
    plt.close()

    print("\n Step 3: Stratifying Splits and Training Class PCA Engines")
    data_train, data_test, y_train, y_test = train_test_split(
        data_mtx, labels, test_size=0.25, stratify=labels, random_state=42
    )
    y_train, y_test = np.array(y_train), np.array(y_test)

    # Global PCA Scree Plot mapping
    pca_global = my_PCA(data_train)
    visualize_var_pcs(pca_global, fig_path=os.path.join(FIGS, 'pca_reconstruction.png'))

    # PCA database visualisation scatter plot
    plot_whole_db_on_2d(pca_global, data_mtx, fig_path=os.path.join(FIGS,
                                                                    "pca_scatter_plots.png"))  # Note that this matrix includes both test and training sets

    # PCA Approximation plots
    display_pca_approx(data_mtx[0], pca_global, fig_path=os.path.join(FIGS, "pca_approximatrion.png"))

    # PCA Feature engineering: Per-class PCA modeling loops
    pca_per_class = []
    nb_pcs_per_class = 5
    for l in sorted(list(label_names.keys())):
        class_subset = data_train[y_train == l]
        pca_per_class.append(my_PCA(class_subset, n_components=nb_pcs_per_class))

    print("\n Step 5: Extracting Localized Structural Features")
    target_canvas = data_train[0].reshape(TARGET_SIZE)
    fig, axes = plt.subplots(2, 2, figsize=(5, 5))
    axes[0, 0].imshow(target_canvas, cmap='gray')
    axes[0, 0].set_title("Target Image", fontsize=8)
    axes[0, 1].imshow(sobel(target_canvas), cmap='gray')
    axes[0, 1].set_title("Sobel Magnitude", fontsize=8)
    axes[1, 0].imshow(sobel_v(target_canvas), cmap='gray')
    axes[1, 0].set_title("Vertical $g_y$", fontsize=8)
    axes[1, 1].imshow(sobel_h(target_canvas), cmap='gray')
    axes[1, 1].set_title("Horizontal $g_x$", fontsize=8)
    for ax in axes.ravel(): ax.axis('off')
    plt.savefig(os.path.join(FIGS, 'sobel_hog.png'), bbox_inches='tight', dpi=150)
    plt.close()

    X_train_pca = compute_pca_features(pca_per_class, data_train, nb_components=nb_pcs_per_class)
    X_train_hog = np.array([compute_hog(img.reshape(TARGET_SIZE)) for img in data_train])
    X_train_combined = np.hstack((X_train_pca, X_train_hog))

    X_test_pca = compute_pca_features(pca_per_class, data_test, nb_components=nb_pcs_per_class)
    X_test_hog = np.array([compute_hog(img.reshape(TARGET_SIZE)) for img in data_test])
    X_test_combined = np.hstack((X_test_pca, X_test_hog))

    print("\n Step 6: Standardizing Features & Optimizing Classifiers")

    # =========================================================================
    # TASK 6: ANALYSIS
    # =========================================================================

    # --- Test 1: Effect of StandardScaler (k=5, PCA+HOG) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_combined)
    X_test_scaled = scaler.transform(X_test_combined)

    knn_scaled = KNeighborsClassifier(n_neighbors=5)
    knn_scaled.fit(X_train_scaled, y_train)
    train_err_scaled = hamming_loss(y_train, knn_scaled.predict(X_train_scaled))
    test_err_scaled = hamming_loss(y_test, knn_scaled.predict(X_test_scaled))

    knn_raw = KNeighborsClassifier(n_neighbors=5)
    knn_raw.fit(X_train_combined, y_train)
    train_err_raw = hamming_loss(y_train, knn_raw.predict(X_train_combined))
    test_err_raw = hamming_loss(y_test, knn_raw.predict(X_test_combined))

    # --- Test 2: Bias-variance tradeoff — k sweep (scaled features) ---
    k_range = range(1, 26)
    train_errs_k, test_errs_k = [], []
    for k in k_range:
        knn_k = KNeighborsClassifier(n_neighbors=k)
        knn_k.fit(X_train_scaled, y_train)
        train_errs_k.append(hamming_loss(y_train, knn_k.predict(X_train_scaled)))
        test_errs_k.append(hamming_loss(y_test, knn_k.predict(X_test_scaled)))

    # --- Test 3: Feature ablation (k=5, scaled) ---
    ablation_configs = {
        'PCA only':  (X_train_pca, X_test_pca),
        'HOG only':  (X_train_hog, X_test_hog),
        'PCA + HOG': (X_train_combined, X_test_combined),
    }
    ablation_results = {}
    for name, (Xtr, Xte) in ablation_configs.items():
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(Xte)
        knn_a = KNeighborsClassifier(n_neighbors=5)
        knn_a.fit(Xtr_s, y_train)
        ablation_results[name] = (
            hamming_loss(y_train, knn_a.predict(Xtr_s)),
            hamming_loss(y_test, knn_a.predict(Xte_s)),
        )

    # --- Plotting: 3 subplots side by side ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    w = 0.35

    # Plot 1: scaler effect
    x1 = np.arange(2)
    axes[0].bar(x1 - w/2, [train_err_scaled, train_err_raw], w, label='Train', color='#a6a6a6', edgecolor='black')
    axes[0].bar(x1 + w/2, [test_err_scaled, test_err_raw],  w, label='Test',  color='#c00000', edgecolor='black')
    axes[0].set_xticks(x1)
    axes[0].set_xticklabels(['With Scaler', 'Without Scaler'])
    axes[0].set_ylabel('Hamming Loss')
    axes[0].set_ylim(0, 1)
    axes[0].set_title('Effect of StandardScaler (k=5)', fontweight='bold')
    axes[0].legend()

    # Plot 2: k sweep
    axes[1].plot(list(k_range), train_errs_k, 'o-', color='#a6a6a6', label='Train')
    axes[1].plot(list(k_range), test_errs_k,  'o-', color='#c00000', label='Test')
    axes[1].set_xlabel('k (neighbors)')
    axes[1].set_ylabel('Hamming Loss')
    axes[1].set_title('Bias-Variance Tradeoff vs k', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # Plot 3: feature ablation
    names = list(ablation_results.keys())
    x3 = np.arange(len(names))
    axes[2].bar(x3 - w/2, [ablation_results[n][0] for n in names], w, label='Train', color='#a6a6a6', edgecolor='black')
    axes[2].bar(x3 + w/2, [ablation_results[n][1] for n in names], w, label='Test',  color='#c00000', edgecolor='black')
    axes[2].set_xticks(x3)
    axes[2].set_xticklabels(names, rotation=10)
    axes[2].set_ylabel('Hamming Loss')
    axes[2].set_ylim(0, 1)
    axes[2].set_title('Feature Comparison (k=5, scaled)', fontweight='bold')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, 'task6_analysis.png'), bbox_inches='tight', dpi=150)
    plt.close()

    # Diagnostic bar chart (baseline = with scaler, k=5, PCA+HOG)
    train_err, test_err = train_err_scaled, test_err_scaled
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(['Training Set Error', 'Testing Generalization Loss'], [train_err, test_err],
            color=['#a6a6a6', '#c00000'], height=0.4, edgecolor='black')
    ax.set_xlim(0, max(train_err, test_err) + 0.1)
    ax.set_title("Model Generalization Diagnostic Gap", fontweight='bold')
    for bar in ax.patches:
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{bar.get_width() * 100:.1f}%', va='center', fontweight='bold')
    plt.savefig(os.path.join(FIGS, 'train_test_error.png'), bbox_inches='tight', dpi=150)
    plt.close()


    # CONFUSION MATRIX FOR KNN
    print("     Calculating and Visualizing the Confusion Matrix for KNN")

    # 1. Générer les prédictions avec le modèle de référence (knn_scaled avec k=5)
    y_pred_knn = knn_scaled.predict(X_test_scaled)

    # 2. Calculer la matrice de confusion
    cm_knn = confusion_matrix(y_test, y_pred_knn)

    # 3. Récupérer les noms des classes dans l'ordre exact attendu par le classifieur
    display_labels_knn = [label_names[c] for c in knn_scaled.classes_]

    # 4. Configurer la visualisation
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=display_labels_knn)

    # On utilise la palette de couleurs "Reds" pour rester cohérent avec vos autres graphiques
    disp_knn.plot(cmap=plt.cm.Reds, ax=ax_cm, xticks_rotation=45)

    plt.title("Matrice de Confusion pour KNN (k=5, PCA+HOG, Scaled)", fontweight='bold')
    plt.tight_layout()

    # 5. Sauvegarder la figure dans le dossier FIGS
    cm_output_path = os.path.join(FIGS, 'confusion_matrix_knn.png')
    plt.savefig(cm_output_path, bbox_inches='tight', dpi=150)
    plt.close()

    # Class distribution histogram: actual vs predicted
    class_names = [label_names[c] for c in sorted(label_names.keys())]
    actual_counts = [np.sum(y_test == c) for c in sorted(label_names.keys())]
    pred_counts   = [np.sum(y_pred_knn == c) for c in sorted(label_names.keys())]

    x_cls = np.arange(len(class_names))
    bar_w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x_cls - bar_w/2, actual_counts, bar_w, label='Actual',    color='#2e75b6', edgecolor='black')
    ax.bar(x_cls + bar_w/2, pred_counts,   bar_w, label='Predicted', color='#c00000', edgecolor='black')
    ax.set_xticks(x_cls)
    ax.set_xticklabels(class_names, rotation=15, ha='right')
    ax.set_ylabel('Count')
    ax.set_title('Class Distribution — Actual vs Predicted (kNN k=5)', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, 'knn_class_distribution.png'), bbox_inches='tight', dpi=150)
    plt.close()

    print("\n Step 6: Saving Random Image Preprocessing Comparisons")
    # Pick 3 random indices from the database
    random.seed(42)
    sample_indices = random.sample(range(len(bw_dogs)), 3)

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))

    for i, idx in enumerate(sample_indices):
        orig = bw_dogs[idx]
        img_cont = resize_and_pad(orig, target_size=TARGET_SIZE, pad_type='continuous')
        img_blur = resize_and_pad(orig, target_size=TARGET_SIZE, pad_type='propagated_blur')

        # Original (before resizing/padding)
        axes[i, 0].imshow(orig, cmap='gray')
        axes[i, 0].set_title(f"Original {i + 1} ({orig.shape[0]}x{orig.shape[1]})", fontsize=8)
        axes[i, 0].axis('off')

        # Continuous Padding
        axes[i, 1].imshow(img_cont, cmap='gray')
        axes[i, 1].set_title(f"Continuous Padding", fontsize=8)
        axes[i, 1].axis('off')

        # Propagated Blur Padding
        axes[i, 2].imshow(img_blur, cmap='gray')
        axes[i, 2].set_title(f"Padding Black", fontsize=8)
        axes[i, 2].axis('off')

    plt.savefig(os.path.join(FIGS, 'preprocessing_comparaisonV2.png'), bbox_inches='tight', dpi=150)
    plt.close()

    print("\n" + "=" * 50)
    print("DIAGNOSTIC RESULTS SUMMARY")
    print("=" * 50)
    print(f" Training Error Rate (Hamming Loss): {train_err * 100:.2f}%")
    print(f" Testing Error Rate (Generalization Loss):  {test_err * 100:.2f}%")
    print("=" * 50)
    print(f"All figures successfully saved inside the '{FIGS}/' folder.")

    print("\n Step 8: Saving Cache Arrays for Script 02")
    import pickle

    # Sauvegarde des matrices d'images (data_train / data_test)
    np.save(os.path.join(DATA_DIR, "X_train_standard.npy"), data_train)
    np.save(os.path.join(DATA_DIR, "X_test_standard.npy"), data_test)

    # Sauvegarde des étiquettes (labels)
    np.save(os.path.join(DATA_DIR, "y_train_standard.npy"), y_train)
    np.save(os.path.join(DATA_DIR, "y_test_standard.npy"), y_test)
    np.save(os.path.join(DATA_DIR, "labels.npy"), labels)

    # Sauvegarde du dictionnaire des noms de classes
    with open(os.path.join(DATA_DIR, "lbl_names.npy"), "wb") as f:
        pickle.dump(label_names, f)

    print("   [SUCCESS] Fichiers .npy générés avec succès dans le dossier courant !")