# ML for CyberSecurity

Projet de détection d'intrusions réseau basé sur l'apprentissage automatique, en utilisant le dataset **NSL-KDD**. Ce projet s'inscrit dans le cadre d'un travail académique en cybersécurité et data science, visant à concevoir un système de détection hiérarchique d’attaques.

---

## **Chapitre 1 – Introduction**

* **1.1 Contexte de la cybersécurité**
  Présentation des enjeux actuels liés à la sécurité des réseaux informatiques.

* **1.2 Importance de la détection d'intrusion réseau**
  Justification de l’usage de techniques d’IA pour détecter des comportements anormaux.

* **1.3 Présentation du projet**
  Vue d’ensemble du système développé, basé sur un pipeline complet de machine learning.

* **1.4 Objectifs du système de détection basé sur les données**
  Identifier efficacement les attaques et les classifier selon leur type.

---

## **Chapitre 2 – Exploration des Données**

* **2.1 Présentation du dataset NSL-KDD**

  * 2.1.1 Description générale
  * 2.1.2 Types d’attaques : DoS, Probe, R2L, U2R
  * 2.1.3 Attributs : 41 features issues du trafic réseau

* **2.2 Analyse exploratoire**

  * 2.2.1 Statistiques descriptives
  * 2.2.2 Répartition des classes (normales vs attaques)
  * 2.2.3 Déséquilibre des classes

* **2.3 Visualisation des données**

  * 2.3.1 Histogrammes des distributions
  * 2.3.2 Matrice de corrélation
  * 2.3.3 Scatter plots pour l'analyse visuelle

---

## **Chapitre 3 – Prétraitement des Données**

* **3.1 Encodage des variables catégorielles** (one-hot ou label encoding)
* **3.2 Normalisation des données** (standardisation pour SVM, DNN, etc.)
* **3.3 Gestion du déséquilibre de classes** via **SVM-SMOTE**
* **3.4 Split du dataset** en ensembles d’entraînement et de test

---

## **Chapitre 4 – Détection d’Intrusion (Hiérarchie en 2 Étapes)**

* **4.1 Étape 1 – Classification binaire : Normal vs Intrusion**

  * 4.1.1 Modèles supervisés : SVM, Random Forest, Naive Bayes...
  * 4.1.2 Utilisation d’un autoencodeur pour la détection non supervisée

* **4.2 Étape 2 – Classification en 4 types d’attaques**

  * 4.2.1 Rééquilibrage du dataset avec SVM-SMOTE
  * 4.2.2 Modèle final : réseau de neurones profond (DNN)

---

## **Chapitre 5 – Évaluation des Modèles**

* **5.1 Métriques d’évaluation**

  * 5.1.1 Précision, rappel, F1-score, accuracy
  * 5.1.2 Comparaison entre macro-F1 et micro-F1

* **5.2 Résultats – Classification binaire**

* **5.3 Résultats – Classification à 4 classes**

* **5.4 Analyse qualitative avec matrices de confusion**

---

## **Chapitre 6 – Conclusion et Perspectives**

* **6.1 Résumé des résultats obtenus**
* **6.2 Avantages du modèle hiérarchique** pour la détection multi-niveau
* **6.3 Limites rencontrées** : déséquilibre des données, complexité des attaques fines
* **6.4 Pistes d’amélioration** : modèles sensibles au coût, architectures avancées

---

## Lancer le projet

```bash
git clone https://github.com/tinbyte/ML-for-CyberSecurity.git
cd ML-for-CyberSecurity
pip install -r requirements.txt
jupyter notebook
```
