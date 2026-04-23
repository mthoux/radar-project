# Importations de haut niveau : uniquement des modules sûrs et sans interface graphique (GUI)
import time
import numpy as np
from multiprocessing import Process, Queue

# -------------------------
# Le code de visualisation est encapsulé dans une fonction pour éviter d'importer
# des modules GUI (comme PyQt5 ou Matplotlib Qt) lors de l'initialisation des processus esclaves.
# -------------------------
def run_visualization(q1, cfg_radar):
    # Importations liées à l'interface graphique effectuées uniquement dans le processus principal
    import warnings
    warnings.simplefilter("ignore", UserWarning)

    from scipy.interpolate import RegularGridInterpolator
    from direct.showbase.ShowBase import ShowBase
    from direct.task import Task

    import matplotlib
    # Définit le backend Matplotlib sur Qt5 pour l'intégration GUI
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-v0_8-dark')

    from panda3d.core import loadPrcFileData
    # Configure Panda3D pour fonctionner sans fenêtre native (mode "headless" pour le moteur 3D)
    loadPrcFileData('', 'window-type none')   
    loadPrcFileData('', 'audio-library-name null')

    from PyQt5 import QtWidgets

    # Importations des utilitaires personnalisés pour le traitement radar
    from streaming_base.visualization.visualization import (
        configure_ax_bf, 
    )
    from streaming_base.utils.utils import cart2pol


    # Classe principale utilisant Panda3D (ShowBase) comme gestionnaire de tâches (Task Manager)
    class MyApp(ShowBase):
        def __init__(self, queue_1, cfg_radar):
            # Initialise l'environnement Panda3D
            ShowBase.__init__(self)
            self.q1 = queue_1  # File d'attente pour recevoir les données BEV
            self.latest_msg = {}  # Stockage du dernier message reçu
            self.msg_count = set() # Suivi des sources de messages

            # Extraction de la configuration radar (angles phi et indices de distance)
            self.phi = cfg_radar["phi"]
            self.r_idxs = cfg_radar["range_idx"]

            # Configuration de la figure Matplotlib en mode projection polaire
            self.fig = plt.figure(figsize=(6, 6))
            self.ax = self.fig.add_subplot(111, projection='polar')
            self.ax.set_ylabel('')
            # Initialisation de l'image de base (Beamforming) sur l'axe polaire
            self.im = configure_ax_bf(self.ax, self.phi, self.r_idxs, 0, 0.3)  

            # Variables pour le calcul des performances (FPS)
            self.last_frame_time = time.time()
            self.frame_counter = 0
            self.fps = 0
            self.last_fps_time = time.time() 

            # Ajout de la fonction de mise à jour à la boucle de rendu de Panda3D
            self.taskMgr.add(self.updateTask, "updateTask")

            # Préparation de la grille cartésienne pour les interpolations
            self.x = np.arange(-cfg_radar["width"], cfg_radar["width"], 1)
            self.y = self.r_idxs
            self.X, self.Y = np.meshgrid(self.x, self.y, indexing='xy')

            # Conversion des coordonnées de la grille en polaire (rayon/angle)
            self.cart2pol = cart2pol(self.X.ravel(), self.Y.ravel()) 

            self.last_artists = []
            num_ticks = 7

            # Génération d'étiquettes radiales (distance) espacées uniformément
            radial_bins = np.linspace(self.r_idxs.min(), self.r_idxs.max(), num_ticks)

            # Conversion des indices de distance en mètres (facteur de conversion 0.045...)
            radial_labels = [f"{rb * 0.045352603795783:.2f}" for rb in radial_bins]

            # Application des graduations et des labels sur l'axe polaire
            self.ax.set_rticks(radial_bins)
            self.ax.set_yticklabels(radial_labels)

        def updateTask(self, task):
            """Tâche récurrente qui vide la file d'attente et met à jour le graphique."""
            try:
                q = self.q1
                # Récupère tous les messages disponibles dans la file (non-bloquant)
                while not q.empty():
                    msg = q.get_nowait()

                    # On vérifie si le message est un dictionnaire (le payload du Processor)
                    if isinstance(msg, dict) and "heatmap" in msg:
                        # On récupère la heatmap depuis la clé "heatmap"
                        self.latest_msg[0] = msg["heatmap"]
                        # Tu peux aussi stocker les tracks pour un usage futur
                        self.latest_tracks = msg.get("tracks", [])
                        self.msg_count.add(0)
                   
            except Exception:
                pass

            # Si de nouvelles données sont arrivées
            if self.msg_count == {0}:
                bf_1 = self.latest_msg[0]

                # Positionnement par défaut du radar (origine 0,0)
                self.x1 = getattr(self, "x1", 0.0)
                self.y1 = getattr(self, "y1", 0.0)

                # Calcul des coordonnées polaires relatives au point (x1, y1)
                phi1 = np.arctan2((self.Y - self.y1).ravel(), (self.X - self.x1).ravel())
                r1 = np.hypot(self.X.ravel() - self.x1, self.Y.ravel() - self.y1)
                cart2pol1 = np.column_stack((phi1, r1))

                # Création d'un interpolateur pour passer des données radar à la grille cartésienne
                interp1 = RegularGridInterpolator(
                    (self.phi, self.r_idxs),
                    bf_1,
                    method='linear', bounds_error=False, fill_value=0
                )
                # Transformation des données radar vers la grille cartésienne
                Z1 = interp1(cart2pol1).reshape(self.X.shape)
                Z_cart = Z1

                # Création d'un second interpolateur pour revenir au format polaire (pour l'affichage)
                interp_cart2pol = RegularGridInterpolator(
                    (self.y, self.x),
                    Z_cart,
                    method='linear',
                    bounds_error=False,
                    fill_value=0
                )

                # Génération des points pour le tracé polaire final
                PHI, R = np.meshgrid(self.phi, self.r_idxs, indexing='ij')
                pts_back = np.column_stack(((R * np.sin(PHI)).ravel(), (R * np.cos(PHI)).ravel()))
                Z_polar = interp_cart2pol(pts_back).reshape(PHI.shape)
                # Retournement de l'axe pour corriger l'orientation de l'affichage
                Z_polar = np.flip(Z_polar, axis=0)

                # Normalisation des données (0 à 1) pour l'affichage des intensités
                to_plot = np.abs(Z_polar)
                mx = np.max(to_plot) if np.max(to_plot) != 0 else 1.0
                to_plot /= mx 

                # Mise à jour des données de l'image Matplotlib sans recréer l'objet
                self.im.set_array(to_plot.ravel()) 

                # Le bloc de calcul FPS est commenté, mais prêt à l'emploi si besoin
                # current_time = time.time()
                # self.frame_counter += 1
                # if current_time - self.last_fps_time >= 1.0:
                #     self.fps = self.frame_counter / (current_time - self.last_fps_time)
                #     self.last_fps_time = current_time
                #     self.frame_counter = 0

                # Actualisation de l'interface graphique
                self.fig.canvas.draw_idle() 
                # Traite les événements Qt (clics, redimensionnement) pour éviter que la fenêtre ne gèle
                QtWidgets.QApplication.processEvents()
                self.msg_count.clear()
                # Petite pause pour laisser la main au système
                plt.pause(0.001)

            # Indique à Panda3D de continuer à exécuter cette tâche au prochain frame
            return Task.cont

    # Instanciation de l'application et lancement de la boucle principale
    app = MyApp(q1, cfg_radar)
    app.run()