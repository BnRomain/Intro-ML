import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# 1. IMPORTATION MODULAIRE DE SCRIPT 01

# On s'assure que le dossier courant est dans le chemin de recherche de Python
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

try:
    # Importation des fonctions clés de votre premier script
    from Script01_PreprocessingExploration import read_and_crop_db, get_resized_db
except ImportError:
    raise ImportError(
        "Impossible de localiser 'Script01_PreprocessingExploration.py'. "
    )

# Configuration des dossiers de sortie
PROJECT_DIR = os.path.join(SCRIPT_DIR, '..')
FIGS_DIR = os.path.join(PROJECT_DIR, 'figures')
os.makedirs(FIGS_DIR, exist_ok=True)


# 2. CONFIGURATION DES PARAMÈTRES

# Choix de l'architecture : 'vgg16' ou 'xception'
ARCHITECTURE = 'vgg16'

if ARCHITECTURE == 'vgg16':
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
    TARGET_SIZE = (224, 224)
elif ARCHITECTURE == 'xception':
    from tensorflow.keras.applications.xception import Xception, preprocess_input
    TARGET_SIZE = (299, 299)

CHOSEN_PAD_TYPE = 'continuous'


# 3. CHARGEMENT ET PRÉPARATION DES DONNÉES (RGB)

print(f"Step 1: Loading dataset in COLOR (RGB) via Script01 utilities...")
_, bw_dogs, labels, label_names = read_and_crop_db(color=True)

cleaned_dogs = []
for img in bw_dogs:
    if img.ndim == 2:  # Si l'image est par hasard en niveaux de gris (H, W) -> (H, W, 3)
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 4:  # Si l'image a un canal Alpha
        # (RGBA) -> on ne garde que (H, W, 3)
        img = img[:, :, :3]
    cleaned_dogs.append(img)
bw_dogs = cleaned_dogs

if bw_dogs is None or len(bw_dogs) == 0:
    print(" [ERREUR] Base de données introuvable. Génération de données factices pour le test...")
    # Simulation de données au format TensorFlow si SmallDB est absent
    X_raw = np.random.randint(0, 255, (100, TARGET_SIZE[0], TARGET_SIZE[1], 3), dtype=np.uint8)
    y = np.random.randint(0, 4, 100)
    label_names = {0: 'Chihuahua', 1: 'Pug', 2: 'Malamute', 3: 'Beagle'}
else:
    print(f" -> Resizing images to {TARGET_SIZE} with '{CHOSEN_PAD_TYPE}' padding...")
    X_resized = get_resized_db(bw_dogs, target_size=TARGET_SIZE, pad_type=CHOSEN_PAD_TYPE)
    X_raw = (X_resized * 255).astype(np.uint8) if X_resized.max() <= 1.0 else X_resized.astype(np.uint8)
    y = np.array(labels)

# Découpage Train/Test
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.25, stratify=y, random_state=42
)

# Application de la fonction de préprocessing spécifique au modèle choisi
# (Gestion de la normalisation et inversion éventuelle des canaux RGB -> BGR pour VGG)
print(" -> Applying model-specific preprocessing...")
X_train = preprocess_input(X_train_raw.astype(np.float32))
X_test = preprocess_input(X_test_raw.astype(np.float32))

NB_CLASSES = len(label_names)


# 4. CONSTRUCTION DU MODÈLE DE TRANSFER LEARNING

print(f"\nStep 2: Building Transfer Learning model architecture ({ARCHITECTURE})...")

if ARCHITECTURE == 'vgg16':
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=TARGET_SIZE + (3,))
else:
    base_model = Xception(weights='imagenet', include_top=False, input_shape=TARGET_SIZE + (3,))

# On gèle les couches convolutives du modèle de base
base_model.trainable = False

# Assemblage de notre classifieur sur mesure
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),  # Réduit la taille des cartes de caractéristiques
    Dense(256, activation='relu'),
    Dropout(0.5),              # Limite le surapprentissage
    Dense(NB_CLASSES, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 5. ENTRAÎNEMENT DU MODÈLE

print("\nStep 3: Training the classification head...")
EPOCHS = 8
BATCH_SIZE = 32

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)


# 6. ÉVALUATION ET MATRICE DE CONFUSION

print("\nStep 4: Evaluating model & computing Confusion Matrix...")

# Obtenir les probabilités de prédiction, puis l'index de la classe max
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Calcul de la matrice
cm = confusion_matrix(y_test, y_pred)
display_labels = [label_names[i] for i in sorted(label_names.keys())]

# Affichage graphique
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)

plt.title(f"Matrice de Confusion pour le Transfer Learning ({ARCHITECTURE.upper()})", fontweight='bold')
plt.tight_layout()

# Sauvegarde de la figure
output_fig_path = os.path.join(FIGS_DIR, f'confusion_matrix_{ARCHITECTURE}.png')
plt.savefig(output_fig_path, bbox_inches='tight', dpi=150)
plt.close()
print(f" [SUCCESS] Graphique sauvegardé avec succès dans : {output_fig_path}")