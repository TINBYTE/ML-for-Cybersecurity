# ML-for-Cybersecurity

# Projet de Détection d'Intrusions Réseau Basée sur l'Apprentissage Automatique (Réplication de "Learning to Detect")

Ce projet vise à mettre en œuvre la méthodologie décrite dans l'article "Learning to Detect: A Data-driven Approach for Network Intrusion Detection" par Zachary Tauscher et al. L'objectif est de développer un système de détection d'intrusions (IDS) en utilisant une approche hiérarchique combinant des modèles d'apprentissage supervisé et non supervisé pour classifier le trafic réseau comme normal ou malveillant, et ensuite identifier le type d'attaque spécifique.

## Étapes de Réalisation du Projet

### 1. Compréhension et Préparation de l'Environnement

*   **1.1. Étude de l'Article de Référence**
    *   Lire attentivement l'article "Learning to Detect: A Data-driven Approach for Network Intrusion Detection" (arXiv:2108.08394v1).
    *   Identifier les sections clés : description du dataset (II.A), visualisation des données (II.B), prétraitement (III.A), méthodologie de détection (III.B, Fig. 4), modèles utilisés, et métriques d'évaluation (IV.A).
*   **1.2. Configuration de l'Environnement de Travail**
    *   Installer Python (version 3.7+ recommandée).
    *   Installer les bibliothèques nécessaires :
        *   `pandas` pour la manipulation des données.
        *   `numpy` pour les opérations numériques.
        *   `scikit-learn` pour les modèles de machine learning classiques, les métriques, et le prétraitement.
        *   `tensorflow` ou `keras` pour les réseaux de neurones (Autoencodeur et DNN).
        *   `matplotlib` et `seaborn` pour la visualisation des données.
        *   `imblearn` pour la technique SVM-SMOTE (si non incluse directement dans scikit-learn, vérifier la version).
*   **1.3. Acquisition du Dataset NSL-KDD**
    *   Télécharger le dataset NSL-KDD. Les fichiers principaux sont `KDDTrain+.txt` et `KDDTest+.txt`.
    *   Se familiariser avec la structure du dataset : 41 caractéristiques et une colonne pour l'étiquette de classe (et une pour le niveau de difficulté, non utilisée directement pour la classification principale dans l'article).

### 2. Exploration et Prétraitement des Données (Data Exploration and Preprocessing)

*   **2.1. Chargement et Inspection Initiale des Données**
    *   Charger les datasets d'entraînement et de test dans des DataFrames pandas.
    *   Attribuer des noms de colonnes significatifs (disponibles avec la documentation du NSL-KDD).
    *   Effectuer une première inspection : `.head()`, `.info()`, `.describe()`.
*   **2.2. Visualisation des Données (Data Visualization)**
    *   Reproduire ou s'inspirer des visualisations de l'article (Fig. 1, 2, 3) :
        *   Histogrammes de distribution pour des caractéristiques spécifiques par type d'attaque.
        *   Carte de corrélation (heatmap) des caractéristiques.
        *   Diagrammes de dispersion (scatter plots) pour des paires de caractéristiques clés.
    *   Objectif : Comprendre les relations entre les caractéristiques et les types d'attaques, identifier les caractéristiques redondantes ou peu informatives.
*   **2.3. Encodage des Caractéristiques Catégorielles**
    *   Identifier les caractéristiques catégorielles : `protocol_type`, `service`, `flag`.
    *   Implémenter l'encodage. L'article mentionne un "LabelCount encoder" (trier les catégories par fréquence). Alternativement, on peut utiliser `OrdinalEncoder` de scikit-learn après avoir trié les catégories par fréquence ou `LabelEncoder` si les valeurs sont simplement converties en chiffres. Pour `service`, qui a beaucoup de modalités, l'approche par fréquence est pertinente.
*   **2.4. Normalisation des Caractéristiques Numériques**
    *   Appliquer la standardisation (Z-score normalization) sur toutes les caractéristiques numériques : `Z = (x - μ) / σ`.
    *   Utiliser `StandardScaler` de scikit-learn. Entraîner le scaler sur le jeu d'entraînement et l'appliquer sur les jeux d'entraînement et de test.
*   **2.5. Préparation des Étiquettes pour les Deux Étapes de Classification**
    *   **Pour la classification binaire :** Créer une étiquette binaire (0 pour 'normal', 1 pour 'attaque' - regroupant tous les types d'attaques).
    *   **Pour la classification multi-classe :** Mapper les types d'attaques spécifiques (DoS, Probe, R2L, U2R) et 'normal' à des entiers (par exemple, 0: Normal, 1: DoS, 2: Probe, 3: R2L, 4: U2R). Pour la deuxième étape, seuls les échantillons d'attaque seront utilisés, avec leurs étiquettes spécifiques.

### 3. Implémentation de la Stratégie de Détection Hiérarchique

*   **3.1. Étape 1 : Classification Binaire (Détection d'Anomalies)**
    *   **3.1.1. Séparation des Données**
        *   Utiliser le jeu d'entraînement prétraité avec les étiquettes binaires.
    *   **3.1.2. Implémentation des Modèles Supervisés**
        *   Entraîner et évaluer les modèles suivants :
            *   Decision Tree (`DecisionTreeClassifier`)
            *   Random Forest (`RandomForestClassifier`)
            *   Naive Bayes (`GaussianNB`, après s'être assuré que les données sont appropriées)
            *   Support Vector Machine (SVM) (`SVC`)
            *   AdaBoost (`AdaBoostClassifier`)
            *   Gradient Boosting (`GradientBoostingClassifier`)
            *   Multi-layer Perceptron (MLP) (`MLPClassifier`)
        *   Utiliser des paramètres par défaut au début, puis envisager une optimisation (ex: `GridSearchCV`).
    *   **3.1.3. Implémentation du Modèle Non Supervisé (Autoencodeur)**
        *   **Architecture (selon l'article) :**
            *   Entrée : nombre de caractéristiques.
            *   Couche d'encodage : par ex. 128 neurones, puis une couche de 64 neurones.
            *   Espace latent (bottleneck) : 15 neurones (selon l'article, Tab II note sur la confusion matrix Fig 5 "with 15 neurons in hidden space").
            *   Couche de décodage : symétrique à l'encodeur, par ex. 64 neurones, puis 128 neurones.
            *   Sortie : nombre de caractéristiques (reconstruction).
            *   Activation : Scaled Exponential Linear Unit (SeLU) pour encodeur/décodeur.
            *   Régularisation : Bruit Gaussien (stddev 0.15) et Dropout (taux 0.05) dans les couches d'encodage.
        *   **Entraînement :**
            *   Entraîner l'Autoencodeur **uniquement** sur les échantillons normaux du jeu d'entraînement.
            *   Fonction de perte : Mean Squared Error (MSE).
            *   Optimiseur : Adam (taux d'apprentissage 0.001).
            *   Paramètres : `batch_size = 32`, 15% des données d'entraînement pour la validation, `early_stopping` (patience 6).
        *   **Détection d'anomalies :**
            *   Calculer l'erreur de reconstruction pour tous les échantillons (normaux et attaques) du jeu de test.
            *   Définir un seuil d'erreur de reconstruction (par exemple, basé sur l'erreur maximale sur les données normales de validation, ou le 95ème percentile, etc.).
            *   Les échantillons avec une erreur supérieure au seuil sont classifiés comme 'attaque'.
*   **3.2. Étape 2 : Classification Multi-classe des Types d'Attaques**
    *   **3.2.1. Préparation des Données pour la Phase 2**
        *   Sélectionner uniquement les échantillons identifiés comme 'attaque' par le meilleur modèle de la phase 1 (ou tous les échantillons d'attaque du dataset original si l'on veut évaluer cette phase indépendamment au début).
        *   Utiliser les étiquettes multi-classes (DoS, Probe, R2L, U2R).
    *   **3.2.2. Gestion du Déséquilibre des Classes avec SVM-SMOTE**
        *   Appliquer SVM-SMOTE sur le jeu d'entraînement de cette phase pour suréchantillonner les classes minoritaires (R2L, U2R).
        *   Utiliser `SVMSMOTE` de la bibliothèque `imblearn`.
    *   **3.2.3. Implémentation du Réseau de Neurones Profond (DNN)**
        *   **Architecture (selon l'article) :**
            *   Entrée : nombre de caractéristiques.
            *   Couche cachée : 1 couche avec 80 neurones, fonction d'activation ReLU.
            *   Couche de sortie : 4 neurones (pour DoS, Probe, R2L, U2R), fonction d'activation Softmax.
        *   **Entraînement :**
            *   Fonction de perte : Cross-entropy (par ex. `categorical_crossentropy` si les étiquettes sont en one-hot, ou `sparse_categorical_crossentropy` si elles sont des entiers).
            *   Optimiseur : Adam (taux d'apprentissage 0.001).
            *   Paramètres : `batch_size = 32`, 15% des données d'entraînement pour la validation, `early_stopping` (patience 6).

### 4. Évaluation des Modèles

*   **4.1. Métriques d'Évaluation (selon l'article, Section IV.A)**
    *   **Pour la classification binaire (Étape 1) :**
        *   Accuracy
        *   Precision
        *   Recall
        *   F1-Score
        *   Matrice de confusion (notamment pour l'Autoencodeur, cf. Fig. 5).
    *   **Pour la classification multi-classe (Étape 2) :**
        *   Accuracy globale.
        *   F1-Score pour chaque classe d'attaque (DoS, Probe, R2L, U2R).
        *   Macro-average F1-Score.
        *   Micro-average F1-Score.
        *   Matrice de confusion (cf. Fig. 6).
*   **4.2. Procédure d'Évaluation**
    *   Évaluer tous les modèles sur le **jeu de test** (`KDDTest+.txt`) prétraité de la même manière que le jeu d'entraînement (mais sans ré-entraîner les scalers ou encodeurs).
    *   Comparer les performances obtenues avec celles reportées dans Table II et Table III de l'article.

### 5. Analyse des Résultats et Conclusion

*   **5.1. Interprétation des Performances**
    *   Analyser les forces et faiblesses de chaque modèle pour chaque tâche.
    *   Discuter de l'impact de SVM-SMOTE sur la classification des classes minoritaires.
    *   Comparer l'approche supervisée et non supervisée (Autoencodeur) pour la détection d'anomalies.
*   **5.2. Rapport et Documentation**
    *   Documenter clairement chaque étape, les choix de conception, les paramètres utilisés.
    *   Présenter les résultats sous forme de tableaux (similaires à ceux de l'article) et de graphiques (matrices de confusion).
*   **5.3. Pistes d'Amélioration et Travaux Futurs (facultatif, pour aller plus loin)**
    *   Optimisation fine des hyperparamètres des modèles.
    *   Exploration d'autres techniques de gestion du déséquilibre.
    *   Utilisation de techniques d'explicabilité des modèles (XAI) pour comprendre les décisions.
    *   Test sur d'autres datasets de détection d'intrusions.