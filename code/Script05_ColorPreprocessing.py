import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import hamming_loss

# Add the code directory to the system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import Script01_PreprocessingExploration as spe

# Set working directory to the code directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def get_normalized_image(img):
    img_float = img.astype(float)
    if np.max(img_float) > 1.0:
        img_float /= 255.0
    return img_float

# 1. Baseline: Average RGB
def extract_average_rgb(img):
    img_float = get_normalized_image(img)
    if len(img_float.shape) == 3:
        return [np.mean(img_float[:, :, 0]), np.mean(img_float[:, :, 1]), np.mean(img_float[:, :, 2])]
    return [np.mean(img_float)] * 3

# 2. Center-Weighted RGB (Filters out background noise near borders)
def extract_center_rgb(img):
    img_float = get_normalized_image(img)
    h, w = img_float.shape[:2]
    # Crop the center 60% of the image (ignoring the outer 20% border)
    y1, y2 = int(h * 0.2), int(h * 0.8)
    x1, x2 = int(w * 0.2), int(w * 0.8)
    center_crop = img_float[y1:y2, x1:x2]
    if len(center_crop.shape) == 3:
        return [np.mean(center_crop[:, :, 0]), np.mean(center_crop[:, :, 1]), np.mean(center_crop[:, :, 2])]
    return [np.mean(center_crop)] * 3

# 3. RGB Color Histograms (Discretizes color intensities into 8 bins per channel)
def extract_color_hist(img, bins=8):
    img_float = get_normalized_image(img)
    if len(img_float.shape) == 3:
        hist_r, _ = np.histogram(img_float[:, :, 0], bins=bins, range=(0, 1))
        hist_g, _ = np.histogram(img_float[:, :, 1], bins=bins, range=(0, 1))
        hist_b, _ = np.histogram(img_float[:, :, 2], bins=bins, range=(0, 1))
        hist = np.concatenate([hist_r, hist_g, hist_b]).astype(float)
        return hist / np.sum(hist)
    else:
        hist_g, _ = np.histogram(img_float, bins=bins, range=(0, 1))
        hist = np.concatenate([hist_g, hist_g, hist_g]).astype(float)
        return hist / np.sum(hist)

print("Loading color dataset...")
imgs, dogs, labels, label_names = spe.read_and_crop_db(color=True)

if imgs is None:
    print("Could not load SmallDB!")
    sys.exit(1)

labels = np.array(labels)

features_dict = {
    'Average RGB (Baseline - 3D)': np.array([extract_average_rgb(img) for img in dogs]),
    'Center-Weighted RGB (Background Filtered - 3D)': np.array([extract_center_rgb(img) for img in dogs]),
    'RGB Color Histograms (Discretized - 24D)': np.array([extract_color_hist(img, bins=8) for img in dogs])
}

results = {}

print("Running preprocessing comparisons...")
for name, X in features_dict.items():
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.25, stratify=labels, random_state=42
    )
    y_train, y_test = np.array(y_train), np.array(y_test)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    
    train_err = hamming_loss(y_train, knn.predict(X_train_scaled))
    test_err = hamming_loss(y_test, knn.predict(X_test_scaled))
    
    results[name] = test_err * 100

print("\n" + "="*55)
print("COLOR PREPROCESSING COMPARISON SUMMARY")
print("="*55)
for name, err in results.items():
    print(f"| {name:<45} | **{err:.2f}%** |")
print("="*55)

# Plot comparison
names = list(results.keys())
errors = list(results.values())

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(names, errors, color=['#c00000', '#2e75b6', '#2ca02c'], height=0.4, edgecolor='black')
ax.set_xlim(0, max(errors) + 10)
ax.set_xlabel('Testing Error Rate (%)')
ax.set_title('Impact of Color Preprocessing on KNN Classification', fontweight='bold')
ax.grid(axis='x', linestyle='--', alpha=0.5)

for bar in bars:
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.2f}%', 
            va='center', fontweight='bold', fontsize=9)

os.makedirs('figures', exist_ok=True)
fig_path = 'figures/color_preprocessing_comparison.png'
plt.savefig(fig_path, bbox_inches='tight', dpi=150)
plt.close()
print(f"Comparison plot saved to '{fig_path}'")
