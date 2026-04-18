# top-level: only safe, non-GUI imports
import time
import numpy as np
from multiprocessing import Process, Queue

from streaming_base.streaming.prod_dca import producer_real_time_1843
from gtrack.config import GTrackConfig2D, PresenceZone2D, Detection
from gtrack.module import GTrackModule2D

RANGE_FACTOR = 0.045352603795783

def build_gtrack_config(cfg_radar) -> GTrackConfig2D:
    return GTrackConfig2D(
        max_points        = 100,
        max_tracks        = 10,
        dt                = cfg_radar.get("dt", 0.05),
        process_noise     = 1.0,
        meas_noise_range  = 0.1,
        meas_noise_az     = 0.02,
        gating_threshold  = 9.21,       
        alloc_range_gate  = 0.5,
        alloc_az_gate     = 0.2,
        alloc_vel_gate    = 1.0,
        min_cluster_points= 2,
        alloc_snr_threshold= 5.0,
        min_snr_threshold = 2.0,
        init_state_cov    = 100.0,
        det_to_active_count= 3,
        det_to_free_count = 3,
        act_to_free_count = 5,
        presence_zones    = [
            PresenceZone2D(x_min=-3, x_max=3, y_min=0, y_max=5)
        ],
        pres_on_count     = 3,
        pres_off_count    = 10,
    )


def run_visualization(q1, cfg_radar, cfg_cfar):
    import warnings
    warnings.simplefilter("ignore", UserWarning)

    from scipy.interpolate import RegularGridInterpolator
    from direct.showbase.ShowBase import ShowBase
    from direct.task import Task

    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-v0_8-dark')

    from panda3d.core import loadPrcFileData
    loadPrcFileData('', 'window-type none')
    loadPrcFileData('', 'audio-library-name null')

    from PyQt5 import QtWidgets

    from streaming_base.visualization.visualization import (
        configure_ax_bf,
        configure_ax_gtrack,
        update_ax_gtrack
    )

    from streaming_base.utils.utils import cart2pol

    class MyApp(ShowBase):
        def __init__(self, queue_1, cfg_radar):
            ShowBase.__init__(self)

            self.q1 = queue_1
            self.latest_msg = {}
            self.msg_count = set()

            self.phi    = cfg_radar["phi"]
            self.r_idxs = cfg_radar["range_idx"]

            self.gtrack = GTrackModule2D(build_gtrack_config(cfg_radar))

            self.fig_1 = plt.figure(figsize=(6, 6))
            self.ax_1 = self.fig_1.add_subplot(111, projection='polar')
            self.im = configure_ax_bf(self.ax_1, self.phi, self.r_idxs, 0, 0.3)

            self.fig_2 = plt.figure(figsize=(6, 6))
            self.ax_2 = self.fig_2.add_subplot(111)

            configure_ax_gtrack(
                self.ax_2,
                cfg_radar["width"],
                len(self.r_idxs)
            )

            self.last_artists = []

            self.x = np.arange(-cfg_radar["width"], cfg_radar["width"], 1)
            self.y = self.r_idxs
            self.X, self.Y = np.meshgrid(self.x, self.y, indexing='xy')

            self.running_max = 1.0

            self.taskMgr.add(self.updateTask, "updateTask")

        def _get_detections(self, Z_polar_raw):
            detections = []
            amp = np.abs(Z_polar_raw)

            thresh = amp.max() * 0.2
            candidates = np.argwhere(amp > thresh)

            for (ai, ri) in candidates:
                angle = self.phi[ai]
                r_bin = self.r_idxs[ri]
                r_m   = r_bin * RANGE_FACTOR
                snr   = float(amp[ai, ri])

                detections.append(Detection(r_m, angle, 0.0, snr))

            return detections

        def updateTask(self, task):
            try:
                while not self.q1.empty():
                    msg = self.q1.get_nowait()
                    if msg[0] == 'bev':
                        self.latest_msg[0] = msg[1]
                        self.msg_count.add(0)
            except:
                pass

            if self.msg_count == {0}:
                bf_1 = self.latest_msg[0]

                phi1 = np.arctan2(self.Y.ravel(), self.X.ravel())
                r1   = np.hypot(self.X.ravel(), self.Y.ravel())
                cart2pol1 = np.column_stack((phi1, r1))

                interp1 = RegularGridInterpolator(
                    (self.phi, self.r_idxs),
                    bf_1,
                    method='linear',
                    bounds_error=False,
                    fill_value=0
                )

                Z_cart = interp1(cart2pol1).reshape(self.X.shape)

                interp_cart2pol = RegularGridInterpolator(
                    (self.y, self.x),
                    Z_cart,
                    method='linear',
                    bounds_error=False,
                    fill_value=0
                )

                PHI, R = np.meshgrid(self.phi, self.r_idxs, indexing='ij')

                pts_back = np.column_stack((
                    (R * np.sin(PHI)).ravel(),
                    (R * np.cos(PHI)).ravel()
                ))

                Z_polar = interp_cart2pol(pts_back).reshape(PHI.shape)
                Z_polar = np.flip(Z_polar, axis=0)

                raw = np.abs(Z_polar)
                self.running_max = max(self.running_max * 0.99, raw.max())
                to_plot = raw / self.running_max


                self.im.set_array(to_plot.ravel())

                detections = self._get_detections(Z_polar)
                gtrack_out = self.gtrack.step(detections)
                tracks = gtrack_out['tracks']

                update_ax_gtrack(self.ax_2, tracks, self.last_artists)

                self.fig_1.canvas.draw_idle()
                self.fig_2.canvas.draw_idle()

                QtWidgets.QApplication.processEvents()

                self.msg_count.clear()
                plt.pause(0.001)

            return Task.cont

    app = MyApp(q1, cfg_radar)
    app.run()


def main(cfg_radar, cfg_cfar):
    q_main_1 = Queue(maxsize=1)

    producer = Process(
        target=producer_real_time_1843,
        args=(q_main_1, cfg_radar, cfg_cfar, 4096, 4098, "192.168.33.30", "192.168.33.180"),
        daemon=True
    )
    producer.start()
    print("Producer started, launching visualization in main process...")

    run_visualization(q_main_1, cfg_radar, cfg_cfar)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        producer.terminate()
        producer.join()
        print("Shutdown complete.")