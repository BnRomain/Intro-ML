import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import hamming_loss

# Add parent path or current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import Script01_PreprocessingExploration as spe

# Set working directory to the code directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def extract_rgb_average(img):
    # Ensure image values are normalized in [0, 1] or [0, 255]
    img_float = img.astype(float)
    # If values are in [0, 255], normalize to [0, 1] for uniform plotting
    if np.max(img_float) > 1.0:
        img_float /= 255.0
        
    if len(img_float.shape) == 3:
        # RGB image (Height, Width, Channels)
        return [np.mean(img_float[:, :, 0]), np.mean(img_float[:, :, 1]), np.mean(img_float[:, :, 2])]
    else:
        # Grayscale image (Height, Width)
        mean_val = np.mean(img_float)
        return [mean_val, mean_val, mean_val]

print("Loading color dataset...")
# Load in color (color=True)
imgs, dogs, labels, label_names = spe.read_and_crop_db(color=True)

if imgs is None:
    print("Could not load SmallDB!")
    sys.exit(1)

# Extract 3D average color features
print("Extracting 3D average RGB features...")
X_color = np.array([extract_rgb_average(img) for img in dogs])
labels = np.array(labels)

# Split into Train / Test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_color, labels, test_size=0.25, stratify=labels, random_state=42
)

# Standardize features for KNN distance calculation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN classifier
k = 5
print(f"Training KNN classifier with K = {k}...")
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)

# Calculate error rates
train_err = hamming_loss(y_train, knn.predict(X_train_scaled))
test_err = hamming_loss(y_test, knn.predict(X_test_scaled))

print("\n" + "="*50)
print("3D COLOR KNN RESULTS SUMMARY")
print("="*50)
print(f" Features: Average R, Average G, Average B (3 Dimensions)")
print(f" KNN Parameter (K): {k}")
print(f" Training Error Rate: {train_err * 100:.2f}%")
print(f" Testing Error Rate (Generalization Loss): {test_err * 100:.2f}%")
print("="*50)

# Create 3D visualization of the color space
print("\nGenerating 3D scatter plot of dog color distribution...")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Unique labels and colors for breeds
unique_labels = np.unique(labels)
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

for lbl, color in zip(unique_labels, colors):
    mask = labels == lbl
    # We plot the unscaled RGB averages so the axis coordinates correspond directly to R, G, B channels
    ax.scatter(
        X_color[mask, 0], X_color[mask, 1], X_color[mask, 2],
        color=color, label=label_names[lbl], alpha=0.7, edgecolors='k', s=40
    )

ax.set_xlabel('Average Red Channel')
ax.set_ylabel('Average Green Channel')
ax.set_zlabel('Average Blue Channel')
ax.set_title('3D Color Space: Breed Distribution based on Average RGB', fontweight='bold')
ax.legend()
ax.grid(True)

# Save figure
os.makedirs('figures', exist_ok=True)
fig_path = 'figures/color_knn_3d_scatter.png'
plt.savefig(fig_path, bbox_inches='tight', dpi=150)
plt.close()
print(f"3D Scatter plot saved to '{fig_path}'")
