# top-level: only safe, non-GUI imports
import time
import numpy as np
from multiprocessing import Process, Queue

from streaming_base.streaming.prod_dca import producer_real_time_1843
from gtrack.config import GTrackConfig2D, PresenceZone2D, Detection
from gtrack.module import GTrackModule2D

RANGE_FACTOR = 0.045352603795783
MIN_SPEED    = 0.3   # m/s — below this a track is considered static

def build_gtrack_config(cfg_radar) -> GTrackConfig2D:
    return GTrackConfig2D(
        max_points         = 100,
        max_tracks         = 10,
        dt                 = cfg_radar.get("dt", 0.05),
        process_noise      = 1.0,
        meas_noise_range   = 0.1,
        meas_noise_az      = 0.02,
        gating_threshold   = 9.21,
        alloc_range_gate   = 0.5,
        alloc_az_gate      = 0.2,
        alloc_vel_gate     = 1.0,
        min_cluster_points = 2,
        alloc_snr_threshold= 5.0,
        min_snr_threshold  = 2.0,
        init_state_cov     = 100.0,
        det_to_active_count= 3,
        det_to_free_count  = 3,
        act_to_free_count  = 5,
        presence_zones     = [
            PresenceZone2D(x_min=-3, x_max=3, y_min=0, y_max=5)
        ],
        pres_on_count      = 3,
        pres_off_count     = 10,
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

    class MyApp(ShowBase):
        def __init__(self, queue_1, cfg_radar):
            ShowBase.__init__(self)
            self.q1        = queue_1
            self.latest_msg = {}
            self.msg_count  = set()

            self.phi    = cfg_radar["phi"]
            self.r_idxs = cfg_radar["range_idx"]

            self.gtrack = GTrackModule2D(build_gtrack_config(cfg_radar))

            # ── Cartesian grid (metres) ──────────────────────────────────────
            r_max_m   = self.r_idxs.max() * RANGE_FACTOR
            x_m       = np.linspace(-r_max_m, r_max_m, 400)
            y_m       = np.linspace(0,         r_max_m, 400)
            self.xg, self.yg = x_m, y_m
            self.Xg,  self.Yg = np.meshgrid(x_m, y_m, indexing='xy')  # (400,400)

            # ── Figure ───────────────────────────────────────────────────────
            self.fig, self.ax = plt.subplots(1, 1, figsize=(7, 7))
            self.ax.set_xlabel("x  (m)")
            self.ax.set_ylabel("y  (m)")
            self.ax.set_xlim(x_m.min(), x_m.max())
            self.ax.set_ylim(y_m.min(), y_m.max())
            self.ax.set_aspect('equal')

            self.im = self.ax.imshow(
                np.zeros((400, 400)),
                extent=[x_m.min(), x_m.max(), y_m.min(), y_m.max()],
                origin='lower', aspect='equal',
                cmap='viridis', vmin=0, vmax=1,
                interpolation='bilinear'
            )

            self.track_scatter = self.ax.scatter(
                [], [], s=140, c='red', marker='o',
                zorder=5, label='moving object'
            )
            self.ax.legend(loc='upper right')

            self.running_max = 1.0

            self.taskMgr.add(self.updateTask, "updateTask")

        def _polar_to_cartesian(self, bf_polar):
            """
            Interpolate polar heatmap (phi × r_idxs) onto the Cartesian grid.
            Returns a (400, 400) float array in metres.
            """
            interp = RegularGridInterpolator(
                (self.phi, self.r_idxs), bf_polar,
                method='linear', bounds_error=False, fill_value=0
            )
            # query points: for every Cartesian (x, y) compute (phi, r_bin)
            phi_q  = np.arctan2(self.Xg.ravel(), self.Yg.ravel())   # azimuth
            r_q    = np.hypot(self.Xg.ravel(), self.Yg.ravel()) / RANGE_FACTOR  # → bins
            pts    = np.column_stack((phi_q, r_q))
            return interp(pts).reshape(self.Xg.shape)

        def _get_detections(self, bf_polar):
            """
            Threshold the polar heatmap and return Detection objects.
            Doppler is unavailable from BEV → 0.0 (gtrack estimates velocity).
            """
            amp    = np.abs(bf_polar)
            thresh = amp.max() * 0.20
            detections = []
            for (ai, ri) in np.argwhere(amp > thresh):
                r_m = self.r_idxs[ri] * RANGE_FACTOR
                detections.append(
                    Detection(r_m, self.phi[ai], 0.0, float(amp[ai, ri]))
                )
            return detections

        def updateTask(self, task):
            # drain queue
            try:
                while not self.q1.empty():
                    msg = self.q1.get_nowait()
                    if msg[0] == 'bev':
                        self.latest_msg[0] = msg[1]
                        self.msg_count.add(0)
            except Exception:
                pass

            if self.msg_count == {0}:
                bf_polar = self.latest_msg[0]

                # ── heatmap ──────────────────────────────────────────────────
                Z_cart = self._polar_to_cartesian(bf_polar)
                raw    = np.abs(Z_cart)
                self.running_max = max(self.running_max * 0.99, raw.max())
                self.im.set_data(raw / self.running_max)

                # ── tracking ─────────────────────────────────────────────────
                detections = self._get_detections(bf_polar)
                gtrack_out = self.gtrack.step(detections)
                tracks     = gtrack_out['tracks']
                presence   = gtrack_out['presence']

                # only plot tracks that are moving
                xs, ys = [], []
                for t in tracks:
                    if t['status'] not in ('DETECTION', 'ACTIVE'):
                        continue
                    vx, vy = t['vel']
                    if np.hypot(vx, vy) < MIN_SPEED:
                        continue
                    tx, ty = t['pos']
                    xs.append(tx)
                    ys.append(ty)
                    print(f"Moving object  x={tx:.2f}m  y={ty:.2f}m  "
                          f"speed={np.hypot(vx,vy):.2f}m/s")

                self.track_scatter.set_offsets(
                    np.column_stack((xs, ys)) if xs else np.empty((0, 2))
                )

                # ── title ────────────────────────────────────────────────────
                if presence:
                    self.ax.set_title("PRESENCE DETECTED", color='red',  fontsize=11)
                else:
                    self.ax.set_title("no presence",       color='gray', fontsize=9)

                self.fig.canvas.draw_idle()
                QtWidgets.QApplication.processEvents()
                self.msg_count.clear()
                plt.pause(0.001)

            return Task.cont

    MyApp(q1, cfg_radar).run()


def main(cfg_radar, cfg_cfar):
    q_main_1 = Queue(maxsize=1)

    producer = Process(
        target=producer_real_time_1843,
        args=(q_main_1, cfg_radar, cfg_cfar, 4096, 4098,
              "192.168.33.30", "192.168.33.180"),
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