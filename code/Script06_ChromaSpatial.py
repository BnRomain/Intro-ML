import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2ycbcr, rgb2gray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import hamming_loss

# Add the code directory to the system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import Script01_PreprocessingExploration as spe

# Set working directory to the code directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Extract Chromatic Spatial Pyramids (CSP) features
# 4x4 Grid, 4 color bins for Cb, 4 color bins for Cr (256 features)
def extract_chroma_spatial_features(img_color, grid_size=4, color_bins=4):
    ycbcr = rgb2ycbcr(img_color)
    cb = ycbcr[:, :, 1]
    cr = ycbcr[:, :, 2]
    
    h, w = img_color.shape[:2]
    cell_h = h // grid_size
    cell_w = w // grid_size
    
    features = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            cell_cb = cb[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            cell_cr = cr[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            
            # Compute 2D histogram of (Cb, Cr)
            hist, _, _ = np.histogram2d(
                cell_cb.ravel(), cell_cr.ravel(), 
                bins=(color_bins, color_bins), 
                range=[[16, 240], [16, 240]]
            )
            
            hist_sum = np.sum(hist)
            if hist_sum > 0:
                hist = hist / hist_sum
                
            features.extend(hist.ravel())
            
    return np.array(features)

print("Loading color dataset...")
imgs_color, dogs_color, labels, label_names = spe.read_and_crop_db(color=True)

if imgs_color is None:
    print("Could not load SmallDB!")
    sys.exit(1)

labels = np.array(labels)

# 1. Preprocess color images using optimal local blur padding
print("Preprocessing images (resizing & padding)...")
TARGET_SIZE = spe.TARGET_SIZE
resized_dogs_color = spe.get_resized_db(dogs_color, target_size=TARGET_SIZE, pad_type='propagated_blur')

# Create grayscale version for PCA and HOG
resized_dogs_gray = np.array([rgb2gray(img) for img in resized_dogs_color])

# 2. Extract Color features (256 dimensions)
print("Extracting 4x4 Chromatic Spatial features...")
X_color_features = np.array([extract_chroma_spatial_features(img) for img in resized_dogs_color])

# 3. Extract baseline PCA + HOG features
data_mtx_gray = spe.convert_ndarrays2data_matrix(resized_dogs_gray)
indices = np.arange(len(labels))
train_idx, test_idx = train_test_split(
    indices, test_size=0.25, stratify=labels, random_state=42
)
y_train, y_test = labels[train_idx], labels[test_idx]

# Train PCA models on train split
pca_per_class = []
nb_pcs_per_class = 5
for l in sorted(list(label_names.keys())):
    class_subset = data_mtx_gray[train_idx][y_train == l]
    pca_per_class.append(spe.my_PCA(class_subset, n_components=nb_pcs_per_class))
    
X_train_pca = spe.compute_pca_features(pca_per_class, data_mtx_gray[train_idx], nb_components=nb_pcs_per_class)
X_test_pca = spe.compute_pca_features(pca_per_class, data_mtx_gray[test_idx], nb_components=nb_pcs_per_class)

# Extract HOG features (4x4 grid)
X_train_hog = np.array([spe.compute_hog(img) for img in resized_dogs_gray[train_idx]])
X_test_hog = np.array([spe.compute_hog(img) for img in resized_dogs_gray[test_idx]])

# 4. Combine Feature matrices
# A: Baseline (PCA + HOG = 148D)
X_train_base = np.hstack((X_train_pca, X_train_hog))
X_test_base = np.hstack((X_test_pca, X_test_hog))

# B: Just Color (CSP = 256D)
X_train_color = X_color_features[train_idx]
X_test_color = X_color_features[test_idx]

# C: Combined (PCA + HOG + Color = 404D)
X_train_combined = np.hstack((X_train_pca, X_train_hog, X_train_color))
X_test_combined = np.hstack((X_test_pca, X_test_hog, X_test_color))

experiments = {
    'PCA + HOG Shape Baseline (148D)': (X_train_base, X_test_base),
    'Just Chromatic Spatial Color (256D)': (X_train_color, X_test_color),
    'PCA + HOG + Color Combined (404D)': (X_train_combined, X_test_combined)
}

results = {}

print("\nEvaluating classifiers...")
for name, (X_tr, X_te) in experiments.items():
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_tr_scaled, y_train)
    
    train_err = hamming_loss(y_train, knn.predict(X_tr_scaled))
    test_err = hamming_loss(y_test, knn.predict(X_te_scaled))
    
    results[name] = test_err * 100
    print(f"| {name:<35} | Test Error: {test_err*100:.2f}% |")

# Plot results
names = list(results.keys())
errors = list(results.values())

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(names, errors, color=['#2e75b6', '#c00000', '#2ca02c'], height=0.4, edgecolor='black')
ax.set_xlim(0, max(errors) + 15)
ax.set_xlabel('Testing Error Rate (%)')
ax.set_title('Evaluation: Shape vs. Chromatic Spatial Color vs. Combined', fontweight='bold')
ax.grid(axis='x', linestyle='--', alpha=0.5)

for bar in bars:
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.2f}%', 
            va='center', fontweight='bold', fontsize=9)

os.makedirs('figures', exist_ok=True)
fig_path = 'figures/chroma_spatial_evaluation.png'
plt.savefig(fig_path, bbox_inches='tight', dpi=150)
plt.close()
print(f"\nEvaluation plot saved to '{fig_path}'")
