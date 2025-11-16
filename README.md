# 🚗 Implémentation "World Model" pour CarRacing (sur Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

Ce projet est une implémentation du concept de "World Model" (Ha & Schmidhuber, 2018) pour entraîner un agent à conduire dans l'environnement CarRacing-v2 de Gymnasium.

L'ensemble du processus est décomposé en 5 notebooks Google Colab, chacun gérant une phase distincte du projet. L'agent final est entraîné **entièrement à l'intérieur d'un "rêve" généré** par un modèle prédictif du monde.

---

## 🏗️ Architecture du Modèle

Le "World Model" n'est pas un seul réseau de neurones, mais une architecture en trois parties :

1.  **CVAE (L'Œil)** : Un Autoencodeur Variationnel Convolutionnel (`vae.pth`).
    * **Rôle :** Apprend à compresser les images brutes du jeu (64x64x3) en un petit vecteur d'état latent $z$ (de 32 dimensions). Il apprend la *perception*.

2.  **MDN-RNN (Le Moteur de Rêve)** : Un Réseau de Neurones Récurrent à Mélange de Densité (`rnn.pth`).
    * **Rôle :** Apprend à prédire le *futur*. En prenant l'état latent $z_t$ et l'action $a_t$, il prédit la distribution de probabilité du prochain état latent $z_{t+1}$. Il apprend les *règles* et la *physique* du monde.

3.  **Controller (L'Agent)** : Un simple réseau de neurones linéaire (`controller.pth`).
    * **Rôle :** C'est le "cerveau" de l'agent. Il prend la perception $z_t$ et la mémoire $h_t$ du RNN pour décider de l'action $a_t$. Il est entraîné à maximiser la récompense *à l'intérieur du rêve* généré par le RNN.

---

## 🗂️ Structure du Projet

Ce projet est conçu pour être exécuté sur Google Drive. Assurez-vous de placer tous les notebooks dans un dossier unique (`World_model`) et de monter votre Drive au début de chaque notebook.

```arborescence
Mon Drive/
└── Colab Notebooks/
    └── World_model/
        │
        ├── 01_collect_data.ipynb      (Phase 1a)
        ├── 02_train_vae.ipynb         (Phase 1b)
        ├── 03_train_rnn.ipynb         (Phase 2)
        ├── 04_train_controller.ipynb  (Phase 3)
        ├── 05_run_agent.ipynb         (Phase 4: Exécution)
        │
        ├── data/
        │   ├── carracing_data.npz   (Sortie de 01)
        │   └── rnn_data.npz         (Sortie de 03)
        │
        ├── videos/
        │   └── rl-video-episode-0.mp4 (Sortie de 05)
        │
        ├── vae.pth                  (Sortie de 02)
        ├── rnn.pth                  (Sortie de 03)
        ├── controller.pth           (Sortie de 04)
        │
        └── README.md                (Ce fichier)
