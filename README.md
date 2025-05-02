# Vision Stéréo - Reconstruction 3D

## 🎯 Objectif

Ce projet propose une solution logicielle complète pour la reconstruction 3D à partir d’un système stéréo de deux caméras USB, avec une interface graphique utilisateur.

## 📦 Fonctionnalités principales

* Calibration des caméras via un damier
* Correction de la distorsion
* Détection et appariement des points clés (SIFT)
* Estimation de la géométrie stéréo (matrice fondamentale et essentielle)
* Triangulation 3D des points
* Visualisation du nuage de points avec Open3D ou un outil externe (MeshLab, CloudCompare)

## 📁 Structure du projet

```
Computer_Vision_App/
├── gui.py                  # Interface graphique principale
├── main.py                 # Lancement de l'application
├── .gitignore            # Ignorer les fichiers inutiles dans Git
├── calibration_module.py  # Module de calibration monoculaire
├── processing.py          # Pipeline de traitement de la reconstruction 3D
├── requirements.txt       # Liste des dépendances Python
├── README.md              # Documentation du projet
├── captures/              # Dossier des images capturées
├── images/                # Dossier avec les images de test
└── reconstruction.ply     # Nuage de points généré
```

## ⚙️ Installation

### 1. Cloner le projet

```bash
git clone https://github.com/thoukam/Computer_Vision_JYV.git
cd Computer_Vision_JYV
```

### 2. Créer un environnement virtuel (optionnel mais recommandé)

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

## 🚀 Lancement de l’application

```bash
python main.py
```

## 🖼️ Visualisation du nuage de points

Après traitement, le fichier `reconstruction.ply` est généré.
Utilisez :

* `meshlab reconstruction.ply`
* ou `cloudcompare reconstruction.ply`

Le logiciel vous demandera quel visualiseur utiliser.

## 🧊 Export exécutable (optionnel)

### Avec PyInstaller :

```bash
pip install pyinstaller
pyinstaller --onefile --windowed gui.py
```

Le binaire sera généré dans `dist/gui(.exe)`.

## ✅ Testé sur

* Ubuntu 22.04
* Windows 11 (avec Python 3.12)

## 🙌 Crédits
Projet réalisé dans le cadre du module Computer Vision – Master 1
Université de Bourgogne - Polytech Dijon
Encadré par Prof. Y. Fougerolle

Étudiants :

    AKOUANGHOU JEOL RUBEN THERENCE

    MENDJE TCHEMMOE VANELLE LETICIA

    THOUKAM THOTCHUM YVES
