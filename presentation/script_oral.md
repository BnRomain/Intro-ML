# Script oral — Classification d'images de chiens

**Durée cible : 9 min 30 (marge avant les 10 min) · 15 slides · 5 min de questions après.**

Répartition : **Evrard** (slides 1-5) · **Romain** (slides 6-10) · **Zouhair** (slides 11-15).
Minutage cumulé indiqué à droite. Les phrases sont à dire, pas à lire mot à mot : appropriez-vous-les.

---

## EVRARD — Intro & préparation des images (~2:50)

### Slide 1 — Titre *(0:20)*
« Bonjour, on va vous présenter notre projet de classification d'images : à partir de la photo d'un chien, reconnaître sa race. Je commence avec les données et leur préparation, Romain enchaînera sur les caractéristiques et le kNN, et Zouhair finira sur le SVM et les perspectives. »

### Slide 2 — Plan *(0:20)*
« Le fil est simple : on part des images brutes, on en extrait des caractéristiques avec la PCA et le HOG, on teste deux classifieurs, le kNN et le SVM, et on termine sur pourquoi on bloque et comment aller plus loin. »

### Slide 3 — Les données *(0:50)*
« On travaille sur SmallDB, un petit jeu tiré du Stanford Dogs Dataset, avec six races : Chihuahua, basset, Kerry blue terrier, groenendael, malinois et chow. Chaque photo vient avec un fichier XML qui indique le cadre du chien.
Avant tout, on vérifie que les classes sont équilibrées, sinon le taux de bonnes réponses serait trompeur. On mesure l'entropie de Shannon de la répartition : on trouve 1,786, juste en dessous du maximum théorique de ln(6), soit 1,792. Le graphe le confirme : chaque race compte entre 150 et 196 images, sans race dominante. Les classes sont donc bien équilibrées, et on peut juger nos modèles au simple taux de bonnes réponses. »

### Slide 4 — Découpage *(0:40)*
« Sur les photos brutes, il y a beaucoup de décor autour du chien. On se sert du cadre donné par le XML pour recadrer sur le chien uniquement. À gauche l'image d'origine, à droite ce qu'on garde. Ça enlève une bonne partie du bruit de fond avant même de calculer quoi que ce soit. »

### Slide 5 — Redimensionnement *(0:50)*
« Les classifieurs ont besoin d'images de même taille : on les ramène toutes en 64 par 64. Mais si on redimensionne brutalement, le chien est écrasé ou étiré. Donc on garde les proportions et on place l'image au centre, ce qui laisse des bords vides à remplir.
On a testé plusieurs remplissages : noir, blanc, continu, miroir ou par propagation de flou. On retient la propagation de flou (propagated blur) : les remplissages noirs ou blancs uniformes créent des transitions brusques sur les bords, ce qui génère des contours artificiels gênant le calcul du HOG. Je passe la main à Romain. »

---

## ROMAIN — Caractéristiques & kNN (~3:30)

### Slide 6 — PCA *(1:00)*
« Merci. Une image en 64 par 64, c'est plus de 4 000 valeurs : beaucoup trop pour travailler directement. La PCA résume cette information sur bien moins d'axes.
Le graphe de gauche montre la variance portée par chaque axe : les premiers concentrent presque tout, on atteint 90 % de l'information avec très peu d'axes. Ça veut dire que les images contiennent énormément de redites.
À droite, on reconstruit une image à partir de ses k premiers axes : avec peu d'axes c'est flou, et plus on en ajoute plus l'image redevient nette, l'écart à l'original tombant vers zéro. »

### Slide 7 — PCA scatter *(0:45)*
« Voici le point le plus important du projet. Quand on projette toutes les images sur les deux premiers axes de la PCA, une couleur par race, les races sont complètement mélangées : aucun groupe net.
C'est un signal très fort, dès le départ : aucune frontière simple ne sépare les races. On verra que c'est ça qui limite tous nos modèles ensuite. »

### Slide 8 — HOG *(0:45)*
« La PCA capte surtout l'allure générale, pas les contours. Pour ça on ajoute le HOG. On découpe l'image en petites cases, et dans chaque case on regarde dans quelles directions vont les contours, calculés avec les filtres de Sobel. On distingue bien la silhouette du chien.
Au final on met bout à bout PCA et HOG : on a à la fois l'allure générale et les contours. »

### Slide 9 — kNN *(1:00)*
« On classe d'abord avec le kNN. Trois observations, de gauche à droite.
À gauche : il faut normaliser. La PCA donne de grands nombres, le HOG de petits ; sans normalisation, ce sont les grands qui décident de tout. Avec un StandardScaler, l'erreur de test baisse.
Au centre : le compromis sur k. À k égal 1, l'erreur d'entraînement est nulle, chaque image est sa propre voisine ; quand k augmente, on lisse, l'erreur d'entraînement monte. Petit k colle aux données, grand k lisse trop.
À droite : PCA et HOG ensemble donnent la meilleure erreur de test, même si le gain reste modeste. »

### Slide 10 — Erreurs du kNN *(0:40)*
« Voilà où notre kNN de référence se trompe, en k égal 5. La matrice de confusion montre qu'il prédit trop souvent certaines races, Kerry blue et chow, et pas assez d'autres, Chihuahua et groenendael. Le modèle penche vers les races qui occupent le plus de place dans l'espace des caractéristiques. C'est encore une conséquence du mélange qu'on a vu. Zouhair, pour le SVM. »

---

## ZOUHAIR — SVM, limites & perspectives (~3:10)

### Slide 11 — SVM *(1:00)*
« Merci. En deuxième partie, on automatise tout le pipeline avec scikit-learn, Pipeline et FeatureUnion, et on remplace le kNN par un SVM.
On vérifie d'abord que train et test ont la même répartition des races : la similarité cosinus vaut 0,99995, donc rien ne fausse l'évaluation.
On règle les paramètres par GridSearch en validation croisée : on retient C égal 10, gamma 5 millièmes, et 5 axes PCA par race.
Le tableau compare les stratégies un-contre-un et un-contre-tous, en linéaire et en RBF. Le RBF bat le linéaire, logique vu que les races ne sont pas séparables par des droites. Le meilleur, un-contre-tous RBF, atteint 59,36 %. »

### Slide 12 — Confusion SVM *(0:35)*
« La matrice de confusion du meilleur SVM montre une diagonale un peu plus marquée que le kNN : certaines races sont bien reconnues. Mais il reste pas mal de confusions, notamment entre le Kerry blue terrier et le groenendael. Comme ils se ressemblent beaucoup, une excellente idée de prolongement serait d'entraîner un classifieur binaire dédié pour cette paire afin de mieux les discriminer. »

### Slide 13 — Pourquoi on plafonne *(0:45)*
« La vraie question : kNN comme SVM tournent autour de 55 à 59 %. Pourquoi ?
Ce n'est ni le classifieur ni le réglage. Le problème, ce sont les caractéristiques. La PCA et le HOG décrivent des choses trop générales, alors que les races se jouent sur des détails fins : la forme des oreilles, le museau, le poil. On le voyait déjà sur le scatter PCA du début. Aucun réglage de k, de C ou de gamma ne peut créer une séparation qui n'existe pas dans les données. »

### Slide 14 — Transfer learning *(0:50)*
« D'où la solution : le transfer learning. On utilise VGG16, un réseau déjà entraîné sur des millions d'images, pour calculer les caractéristiques à la place de la PCA et du HOG. On gèle ses couches et on entraîne juste une petite tête sur nos six races, avec les images en couleur en 224 par 224.
Et le résultat est sans appel : on passe de 59 % à 98 % de bonnes réponses, seulement 5 erreurs sur 251 images. La matrice de confusion est quasi diagonale, alors que celles du kNN et du SVM étaient très confuses. Ça confirme tout notre propos : le problème, c'était les caractéristiques, pas le classifieur. À noter quand même que ces six races font partie des classes d'ImageNet, ce qui aide clairement VGG16. »

### Slide 15 — Conclusion *(0:30)*
« Pour conclure : on a construit une chaîne complète, du nettoyage des photos à l'évaluation, en passant par les caractéristiques et deux classifieurs. On retient l'utilité de normaliser, le compromis biais-variance, l'importance d'un découpage équilibré, et surtout que la qualité des caractéristiques fixe la limite : environ 59 % en classique, mais 98 % avec le transfer learning. C'est la vraie leçon du projet : changer de caractéristiques compte plus que changer de classifieur. Merci de votre attention, on est prêts pour vos questions. »

---

## Questions probables du jury — réponses prêtes

- **Pourquoi l'entropie plutôt que juste compter ?** Une seule mesure qui résume l'équilibre, comparable au max ln(6) ; pratique pour dire « équilibré » d'un chiffre.
- **Pourquoi la propagation de flou et pas le noir/blanc uniforme ?** Un padding uniforme noir ou blanc crée une coupure nette au bord de l'image. Cette frontière franche génère des gradients artificiels très forts lors du calcul avec les filtres Sobel, ce que le HOG interprète comme de vrais contours visuels. La propagation de flou permet de lisser ces discontinuités de gradients pour ne pas bruiter les descripteurs.

- **k=1 donne erreur d'entraînement nulle, c'est bien ?** Non, c'est mécanique : chaque point est son propre voisin. Seule l'erreur de test compte. Idem avec weights='distance' : erreur d'entraînement toujours nulle, c'est un artefact.
- **Pourquoi RBF > linéaire ?** Les classes ne sont pas linéairement séparables (cf. scatter PCA) ; le noyau RBF gère des frontières courbes.
- **OvO vs OvR ?** OvO entraîne un SVM par paire de classes, OvR un SVM par classe contre le reste. Ici OvR RBF gagne de peu (59,36 %).
- **Pourquoi pas plus d'axes PCA / un autre noyau ?** On a fait une GridSearch ; au-delà ça ne décolle pas, parce que la limite vient des caractéristiques, pas du modèle.
- **Transfer learning, pourquoi 98 % ?** Les caractéristiques d'un réseau pré-entraîné encodent des détails visuels appris sur des millions d'images, bien plus discriminants que PCA/HOG. Et ces six races sont des classes d'ImageNet, donc VGG16 les avait déjà apprises, ce qui explique un score aussi haut.
- **N'est-ce pas du surapprentissage / test = validation ?** On évalue sur le test stratifié non vu à l'entraînement ; le val_set du fit est ce même test, donc le 98 % est bien une accuracy de test. La tête entraînée est petite (Dropout 0.5), peu de risque de surapprentissage.
