import time
import numpy as np
import warnings

# On garde les imports GUI à l'intérieur de run_visualization pour éviter 
# les conflits de processus sur Windows/Linux
def run_visualization(q_results, cfg_radar):
    warnings.simplefilter("ignore", UserWarning)

    from scipy.interpolate import RegularGridInterpolator
    from direct.showbase.ShowBase import ShowBase
    from direct.task import Task
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    from PyQt5 import QtWidgets
    from panda3d.core import loadPrcFileData

    # Désactiver la fenêtre 3D native de Panda3D car on utilise Matplotlib
    loadPrcFileData('', 'window-type none')
    loadPrcFileData('', 'audio-library-name null')

    from streaming_base.visualization.visualization import configure_ax_bf
    from streaming_base.utils.utils import cart2pol

    class RadarVisualizer(ShowBase):
        def __init__(self, q_results, cfg_radar):
            ShowBase.__init__(self)
            self.q_results = q_results
            
            # Paramètres radar
            self.phi = cfg_radar["phi"]
            self.r_idxs = cfg_radar["range_idx"]
            self.res_distance = 0.04535  # Ta résolution en mètres

            # Setup Graphique (Polar Plot)
            self.fig = plt.figure(figsize=(7, 7))
            self.ax = self.fig.add_subplot(111, projection='polar')
            plt.style.use('seaborn-v0_8-dark')
            
            # L'image de fond (Heatmap)
            self.im = configure_ax_bf(self.ax, self.phi, self.r_idxs, 0, 0.3)
            
            # Couche supplémentaire : Les points de Tracking (Multi-track)
            # On dessine des points rouges pour les cibles détectées
            self.track_plot = self.ax.scatter([], [], c='red', s=50, edgecolors='white', zorder=5)

            # Grille pour l'interpolation (pré-calculée pour gagner du temps)
            self.x_grid = np.arange(-cfg_radar["width"], cfg_radar["width"], 1)
            self.y_grid = self.r_idxs
            self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid, indexing='xy')
            
            # Ticks (Labels de distance)
            radial_bins = np.linspace(self.r_idxs.min(), self.r_idxs.max(), 7)
            radial_labels = [f"{rb * self.res_distance:.2f}m" for rb in radial_bins]
            self.ax.set_rticks(radial_bins)
            self.ax.set_yticklabels(radial_labels)

            # Lancement de la boucle de rafraîchissement
            self.taskMgr.add(self.updateTask, "updateTask")

        def updateTask(self, task):
            payload = None
            
            # 1. On vide la queue pour ne garder que le résultat le plus récent (Real-time)
            try:
                while not self.q_results.empty():
                    payload = self.q_results.get_nowait()
            except:
                pass

            if payload:
                # Extraction des données envoyées par le Processor
                heatmap_raw = payload["heatmap"]
                tracks = payload["tracks"] # Ta future liste d'objets

                # 2. TRANSFORMATION GRAPHIQUE (Interpolation)
                # On réutilise ta logique pour lisser l'affichage
                interp1 = RegularGridInterpolator(
                    (self.phi, self.r_idxs), heatmap_raw,
                    method='linear', bounds_error=False, fill_value=0
                )
                
                # Conversion Cartésien -> Polaire pour l'affichage final
                PHI, R = np.meshgrid(self.phi, self.r_idxs, indexing='ij')
                pts_back = np.column_stack(((R * np.sin(PHI)).ravel(), (R * np.cos(PHI)).ravel()))
                
                # Note: On simplifie ici le double-interp pour la fluidité
                Z_polar = interp1(np.column_stack((PHI.ravel(), R.ravel()))).reshape(PHI.shape)
                to_plot = np.abs(np.flip(Z_polar, axis=0))
                
                # Normalisation
                mx = to_plot.max() if to_plot.max() != 0 else 1.0
                to_plot /= mx

                # 3. MISE À JOUR DE L'IMAGE
                self.im.set_array(to_plot.ravel())

                # 4. MISE À JOUR DU TRACKING
                if len(tracks) > 0:
                    # Si tu as des tracks : [{'r': 2.0, 'theta': 0.5}, ...]
                    r_coords = [t['r'] for t in tracks]
                    theta_coords = [t['theta'] for t in tracks]
                    self.track_plot.set_offsets(np.c_[theta_coords, r_coords])
                else:
                    # Sinon, on vide les points rouges
                    self.track_plot.set_offsets(np.empty((0, 2)))

                # 5. RENDU
                self.fig.canvas.draw_idle()
                QtWidgets.QApplication.processEvents()
                plt.pause(0.001)

            return Task.cont

    # Lancement de l'application
    app = RadarVisualizer(q_results, cfg_radar)
    app.run()