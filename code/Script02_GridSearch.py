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

print("Loading dataset...")
bw_imgs, bw_dogs, labels, label_names = spe.read_and_crop_db(color=False)

if bw_imgs is None:
    print("Could not load SmallDB!")
    sys.exit(1)

# Grid parameters
target_sizes = [32, 48, 64, 80]
hog_grid_sizes = [2, 4, 6, 8]

results = []

print("Starting grid search...")
for target_size in target_sizes:
    t_size = (target_size, target_size)
    
    # 1. Resize and pad all dog images using propagated_blur
    resized_dogs = np.zeros((len(bw_dogs), target_size, target_size))
    for idx, img in enumerate(bw_dogs):
        resized_dogs[idx] = spe.resize_and_pad(img, target_size=t_size, pad_type='propagated_blur')
        
    data_mtx = spe.convert_ndarrays2data_matrix(resized_dogs)
    
    # 2. Split dataset
    data_train, data_test, y_train, y_test = train_test_split(
        data_mtx, labels, test_size=0.25, stratify=labels, random_state=42
    )
    y_train, y_test = np.array(y_train), np.array(y_test)
    
    # Train PCA models
    pca_per_class = []
    nb_pcs_per_class = 5
    for l in sorted(list(label_names.keys())):
        class_subset = data_train[y_train == l]
        pca_per_class.append(spe.my_PCA(class_subset, n_components=nb_pcs_per_class))
        
    X_train_pca = spe.compute_pca_features(pca_per_class, data_train, nb_components=nb_pcs_per_class)
    X_test_pca = spe.compute_pca_features(pca_per_class, data_test, nb_components=nb_pcs_per_class)
    
    for hog_cells in hog_grid_sizes:
        print(f"Evaluating: Target Size = {target_size}x{target_size} | HOG Grid = {hog_cells}x{hog_cells}...")
        
        # Calculate HOG features with the specific cells
        X_train_hog = np.array([spe.compute_hog(img.reshape(t_size), nb_height_cells=hog_cells, nb_width_cells=hog_cells) for img in data_train])
        X_test_hog = np.array([spe.compute_hog(img.reshape(t_size), nb_height_cells=hog_cells, nb_width_cells=hog_cells) for img in data_test])
        
        # Combine features
        X_train_combined = np.hstack((X_train_pca, X_train_hog))
        X_test_combined = np.hstack((X_test_pca, X_test_hog))
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_combined)
        X_test_scaled = scaler.transform(X_test_combined)
        
        # Train classifier
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train_scaled, y_train)
        
        # Error computation
        train_err = hamming_loss(y_train, knn.predict(X_train_scaled))
        test_err = hamming_loss(y_test, knn.predict(X_test_scaled))
        
        results.append({
            'target_size': target_size,
            'hog_cells': hog_cells,
            'train_error': train_err,
            'test_error': test_err
        })

# Print results as Markdown Table
print("\n" + "="*50)
print("GRID SEARCH RESULTS SUMMARY")
print("="*50)
print("| Target Image Size | HOG Cell Grid | Training Error | Testing Error |")
print("| :---: | :---: | :---: | :---: |")
for res in sorted(results, key=lambda x: x['test_error']):
    print(f"| {res['target_size']}x{res['target_size']} | {res['hog_cells']}x{res['hog_cells']} | {res['train_error']*100:.2f}% | {res['test_error']*100:.2f}% |")
print("="*50)

# Plot results
fig, ax = plt.subplots(figsize=(8, 5))
for hog_cells in hog_grid_sizes:
    x = [res['target_size'] for res in results if res['hog_cells'] == hog_cells]
    y = [res['test_error'] * 100 for res in results if res['hog_cells'] == hog_cells]
    ax.plot(x, y, marker='o', label=f'HOG Grid {hog_cells}x{hog_cells}')

ax.set_xlabel('Target Image Canvas Size')
ax.set_ylabel('Testing Error Rate (%)')
ax.set_title('Grid Search: Target Size vs. HOG Grid (Propagated Blur Padding)')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)

# Save figure
os.makedirs('figures', exist_ok=True)
fig_path = 'figures/grid_search_results.png'
plt.savefig(fig_path, bbox_inches='tight', dpi=150)
plt.close()
print(f"Results plot saved to '{fig_path}'")
