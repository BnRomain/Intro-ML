import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import hamming_loss
from skimage.filters import gaussian

# Add parent path or current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import Script01_PreprocessingExploration as spe

# Set working directory to the code directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def pad_propagated_blur_custom(resized_img, pad_width, iterations, sigma, multi_channel=False):
    padded = np.pad(resized_img, pad_width, mode='edge').astype(float)
    top, bottom_pad = pad_width[0]
    left, right_pad = pad_width[1]
    height, width = padded.shape[:2]
    bottom = height - bottom_pad
    right = width - right_pad
    
    mask = np.ones(padded.shape[:2], dtype=bool)
    mask[top:bottom, left:right] = False
    
    if multi_channel:
        mask_3d = np.repeat(mask[:, :, np.newaxis], padded.shape[2], axis=2)
    else:
        mask_3d = mask
        
    for _ in range(iterations):
        if multi_channel:
            blurred = gaussian(padded, sigma=sigma, channel_axis=2)
        else:
            blurred = gaussian(padded, sigma=sigma)
        padded[mask_3d] = blurred[mask_3d]
        
    return padded

print("Loading dataset...")
bw_imgs, bw_dogs, labels, label_names = spe.read_and_crop_db(color=False)

if bw_imgs is None:
    print("Could not load SmallDB!")
    sys.exit(1)

# Fixed parameters
TARGET_SIZE = spe.TARGET_SIZE
HOG_CELLS = 4

# Hyperparameter grid for the blur
iterations_list = [1, 2, 5, 8, 12, 18]
sigma_list = [0.1, 0.3, 0.5, 0.8, 1.2, 1.8, 2.5]

# To store the testing errors
error_matrix = np.zeros((len(iterations_list), len(sigma_list)))
results = []

print("Starting blur parameter optimization...")

for i, iterations in enumerate(iterations_list):
    for j, sigma in enumerate(sigma_list):
        print(f"Evaluating: iterations={iterations}, sigma={sigma}...")
        
        # Resize and pad with custom blur parameters
        resized_dogs = np.zeros((len(bw_dogs), TARGET_SIZE[0], TARGET_SIZE[1]))
        for idx, img in enumerate(bw_dogs):
            h, w = img.shape[:2]
            height, width = TARGET_SIZE
            h_r = height / h
            w_r = width / w
            if h_r < w_r:
                new_h, new_w = height, int(w * h_r)
                top, bottom = 0, height
                left, right = (width - new_w) // 2, (width - new_w) // 2 + new_w
            else:
                new_h, new_w = int(h * w_r), width
                top, bottom = (height - new_h) // 2, (height - new_h) // 2 + new_h
                left, right = 0, width

            resized_img = spe.img_resize(img, (new_h, new_w))
            pad_h = (top, height - bottom)
            pad_w = (left, width - right)
            
            resized_dogs[idx] = pad_propagated_blur_custom(
                resized_img, (pad_h, pad_w), iterations=iterations, sigma=sigma
            )
            
        data_mtx = spe.convert_ndarrays2data_matrix(resized_dogs)
        
        # Split
        data_train, data_test, y_train, y_test = train_test_split(
            data_mtx, labels, test_size=0.25, stratify=labels, random_state=42
        )
        y_train, y_test = np.array(y_train), np.array(y_test)
        
        # Train class PCA
        pca_per_class = []
        nb_pcs_per_class = 5
        for l in sorted(list(label_names.keys())):
            class_subset = data_train[y_train == l]
            pca_per_class.append(spe.my_PCA(class_subset, n_components=nb_pcs_per_class))
            
        X_train_pca = spe.compute_pca_features(pca_per_class, data_train, nb_components=nb_pcs_per_class)
        X_train_hog = np.array([spe.compute_hog(img.reshape(TARGET_SIZE), nb_height_cells=HOG_CELLS, nb_width_cells=HOG_CELLS) for img in data_train])
        X_train_combined = np.hstack((X_train_pca, X_train_hog))
        
        X_test_pca = spe.compute_pca_features(pca_per_class, data_test, nb_components=nb_pcs_per_class)
        X_test_hog = np.array([spe.compute_hog(img.reshape(TARGET_SIZE), nb_height_cells=HOG_CELLS, nb_width_cells=HOG_CELLS) for img in data_test])
        X_test_combined = np.hstack((X_test_pca, X_test_hog))
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_combined)
        X_test_scaled = scaler.transform(X_test_combined)
        
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train_scaled, y_train)
        
        train_err = hamming_loss(y_train, knn.predict(X_train_scaled))
        test_err = hamming_loss(y_test, knn.predict(X_test_scaled))
        
        error_matrix[i, j] = test_err * 100
        results.append({
            'iterations': iterations,
            'sigma': sigma,
            'train_error': train_err,
            'test_error': test_err
        })

# Print top results
print("\n" + "="*50)
print("TOP 10 BLUR PARAMETER COMBINATIONS")
print("="*50)
print("| Iterations | Sigma (Blur Scale) | Training Error | Testing Error |")
print("| :---: | :---: | :---: | :---: |")
for res in sorted(results, key=lambda x: x['test_error'])[:10]:
    print(f"| {res['iterations']} | {res['sigma']} | {res['train_error']*100:.2f}% | **{res['test_error']*100:.2f}%** |")
print("="*50)

# Plot Heatmap in pure matplotlib
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(error_matrix, cmap='YlOrRd')

# Add annotations
for i in range(len(iterations_list)):
    for j in range(len(sigma_list)):
        val = error_matrix[i, j]
        # Text color dynamic (black or white) based on background shade
        color = "white" if val > np.mean(error_matrix) else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontweight='bold', fontsize=9)

ax.set_xticks(np.arange(len(sigma_list)))
ax.set_yticks(np.arange(len(iterations_list)))
ax.set_xticklabels(sigma_list)
ax.set_yticklabels(iterations_list)

plt.colorbar(im, label='Testing Error Rate (%)')
plt.xlabel('Sigma (Gaussian Blur Scale)')
plt.ylabel('Iterations (Diffusion Depth)')
plt.title('Propagated Blur Optimization Heatmap (Image: 64x64, HOG: 4x4)')

# Save
os.makedirs('figures', exist_ok=True)
fig_path = 'figures/blur_optimization_heatmap.png'
plt.savefig(fig_path, bbox_inches='tight', dpi=150)
plt.close()
print(f"Heatmap saved to '{fig_path}'")
