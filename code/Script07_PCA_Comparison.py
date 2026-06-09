import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2ycbcr, rgb2gray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import hamming_loss
from sklearn.decomposition import PCA

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

# Extract HOG features (4x4 grid)
X_train_hog = np.array([spe.compute_hog(img) for img in resized_dogs_gray[train_idx]])
X_test_hog = np.array([spe.compute_hog(img) for img in resized_dogs_gray[test_idx]])

# A. Extract Global PCA features (20 components)
print("Training Global PCA on shape features...")
pca_global_model = PCA(n_components=20)
X_train_pca_global = pca_global_model.fit_transform(data_mtx_gray[train_idx])
X_test_pca_global = pca_global_model.transform(data_mtx_gray[test_idx])

# B. Extract Class-wise PCA features (5 components per class = 20 components)
print("Training Class-wise PCA on shape features...")
pca_per_class = []
nb_pcs_per_class = 5
for l in sorted(list(label_names.keys())):
    class_subset = data_mtx_gray[train_idx][y_train == l]
    pca_per_class.append(spe.my_PCA(class_subset, n_components=nb_pcs_per_class))
    
X_train_pca_class = spe.compute_pca_features(pca_per_class, data_mtx_gray[train_idx], nb_components=nb_pcs_per_class)
X_test_pca_class = spe.compute_pca_features(pca_per_class, data_mtx_gray[test_idx], nb_components=nb_pcs_per_class)

# Compress Color features to 5 components using PCA
print("Compressing color features using PCA (5 components)...")
pca_color_model = PCA(n_components=5)
X_train_color_pca = pca_color_model.fit_transform(X_color_features[train_idx])
X_test_color_pca = pca_color_model.transform(X_color_features[test_idx])

# Evaluate configurations
configs = {
    'Global PCA + HOG (Baseline)': np.hstack((X_train_pca_global, X_train_hog)),
    'Class-wise PCA + HOG (Baseline)': np.hstack((X_train_pca_class, X_train_hog)),
    'Global PCA + HOG + Color PCA': np.hstack((X_train_pca_global, X_train_hog, X_train_color_pca)),
    'Class-wise PCA + HOG + Color PCA': np.hstack((X_train_pca_class, X_train_hog, X_train_color_pca))
}

results = {}

for name, X_tr in configs.items():
    # Construct corresponding test set
    if 'Global PCA' in name:
        pca_te = X_test_pca_global
    else:
        pca_te = X_test_pca_class
        
    if 'Color PCA' in name:
        X_te = np.hstack((pca_te, X_test_hog, X_test_color_pca))
    else:
        X_te = np.hstack((pca_te, X_test_hog))
        
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_tr_sc, y_train)
    
    test_err = hamming_loss(y_test, knn.predict(X_te_sc))
    results[name] = test_err * 100

print("\n" + "="*70)
print("PCA COMPARISON RESULTS SUMMARY")
print("="*70)
for name, err in results.items():
    print(f"| {name:<35} | Test Error: {err:.2f}% |")
print("="*70)

# Plot comparison chart
names = list(results.keys())
errors = list(results.values())

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.barh(names, errors, color=['#c00000', '#2e75b6', '#ff7f0e', '#2ca02c'], height=0.4, edgecolor='black')
ax.set_xlim(0, max(errors) + 15)
ax.set_xlabel('Testing Error Rate (%)')
ax.set_title('Evaluation: Global vs. Class-wise PCA & Color Compression', fontweight='bold')
ax.grid(axis='x', linestyle='--', alpha=0.5)

for bar in bars:
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.2f}%', 
            va='center', fontweight='bold', fontsize=9)

os.makedirs('figures', exist_ok=True)
fig_path = 'figures/pca_comparison_evaluation.png'
plt.savefig(fig_path, bbox_inches='tight', dpi=150)
plt.close()
print(f"Evaluation plot saved to '{fig_path}'")
