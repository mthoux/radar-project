# top-level: only safe, non-GUI imports
import time
import numpy as np
from multiprocessing import Process, Queue

from streaming_base.streaming.prod_dca import producer_real_time_1843
from src.gtrack.config import GTrackConfig2D, PresenceZone2D, Detection
from src.gtrack.module import GTrackModule2D

RANGE_FACTOR = 0.045352603795783

def build_gtrack_config(cfg_radar) -> GTrackConfig2D:
    return GTrackConfig2D(
        max_points        = 100,
        max_tracks        = 10,
        dt                = cfg_radar.get("dt", 0.05),
        process_noise     = 1.0,
        meas_noise_range  = 0.1,
        meas_noise_az     = 0.02,
        gating_threshold  = 9.21,       # chi2, 2-DOF, 99%
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

    from streaming_base.visualization.visualization import configure_ax_bf
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

            self.fig, self.ax = plt.subplots(1, 1, figsize=(7, 7),
                                             subplot_kw={'projection': 'polar'})
            self.ax.set_ylabel('')
            self.im = configure_ax_bf(self.ax, self.phi, self.r_idxs, 0, 0.3)

            self.track_scatter = self.ax.scatter(
                [], [], s=120, c='red', marker='o',
                zorder=5, label='tracks'
            )
            self.ax.legend(loc='upper right')

            num_ticks = 7
            radial_bins   = np.linspace(self.r_idxs.min(), self.r_idxs.max(), num_ticks)
            radial_labels = [f"{rb * RANGE_FACTOR:.2f}m" for rb in radial_bins]
            self.ax.set_rticks(radial_bins)
            self.ax.set_yticklabels(radial_labels)

            self.x = np.arange(-cfg_radar["width"], cfg_radar["width"], 1)
            self.y = self.r_idxs
            self.X, self.Y = np.meshgrid(self.x, self.y, indexing='xy')
            self.cart2pol  = cart2pol(self.X.ravel(), self.Y.ravel())

            self.running_max = 1.0

            self.taskMgr.add(self.updateTask, "updateTask")

        def _get_detections(self, Z_polar_raw):
            """
            Turn the polar heatmap into a list of Detection objects.
            We threshold by SNR proxy (raw amplitude) and feed peaks to gtrack.
            """
            detections = []
            amp = np.abs(Z_polar_raw)

            thresh = amp.max() * 0.20
            candidates = np.argwhere(amp > thresh)
            for (ai, ri) in candidates:
                angle  = self.phi[ai]
                r_bin  = self.r_idxs[ri]
                r_m    = r_bin * RANGE_FACTOR
                snr    = float(amp[ai, ri])
                # doppler not available from BEV map → 0
                detections.append(Detection(r_m, angle, 0.0, snr))
            return detections

        def updateTask(self, task):
            try:
                q = self.q1
                while not q.empty():
                    msg = q.get_nowait()
                    if msg[0] == 'bev':
                        self.latest_msg[0] = msg[1]
                        self.msg_count.add(0)
            except Exception:
                pass

            if self.msg_count == {0}:
                bf_1 = self.latest_msg[0]

                self.x1 = getattr(self, "x1", 0.0)
                self.y1 = getattr(self, "y1", 0.0)

                phi1 = np.arctan2((self.Y - self.y1).ravel(), (self.X - self.x1).ravel())
                r1   = np.hypot(self.X.ravel() - self.x1, self.Y.ravel() - self.y1)
                cart2pol1 = np.column_stack((phi1, r1))

                interp1 = RegularGridInterpolator(
                    (self.phi, self.r_idxs), bf_1,
                    method='linear', bounds_error=False, fill_value=0
                )
                Z_cart = interp1(cart2pol1).reshape(self.X.shape)

                interp_cart2pol = RegularGridInterpolator(
                    (self.y, self.x), Z_cart,
                    method='linear', bounds_error=False, fill_value=0
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

                detections  = self._get_detections(Z_polar)
                gtrack_out  = self.gtrack.step(detections)
                tracks      = gtrack_out['tracks']
                presence    = gtrack_out['presence']

                self.im.set_array(to_plot.ravel())

                if tracks:
                    azimuths   = []
                    range_bins = []
                    for t in tracks:
                        if t['status'] not in ('DETECTION', 'ACTIVE'):
                            continue
                        tx, ty = t['pos']
                        r_m  = np.hypot(tx, ty)
                        az   = np.arctan2(tx, ty)  
                        r_bin = r_m / RANGE_FACTOR
                        azimuths.append(az)
                        range_bins.append(r_bin)
                        vx, vy = t['vel']
                        speed  = np.hypot(vx, vy)
                        if speed > 0.1:   
                            print(f"There is a moving object at x={tx:.2f}m, y={ty:.2f}m  (speed={speed:.2f}m/s)")
                    self.track_scatter.set_offsets(
                        np.column_stack((azimuths, range_bins)) if azimuths else np.empty((0, 2))
                    )
                else:
                    self.track_scatter.set_offsets(np.empty((0, 2)))

                if presence:
                    self.ax.set_title("PRESENCE DETECTED", color='red', fontsize=11)
                else:
                    self.ax.set_title("no presence", color='gray', fontsize=9)

                self.fig.canvas.draw_idle()
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