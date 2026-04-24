# top-level: only safe, non-GUI imports
import time
import numpy as np
from multiprocessing import Process, Queue

from streaming_base.streaming.prod_dca import producer_real_time_1843

# Conversion factor from range bin index to meters
RANGE_FACTOR = 0.045352603795783

def run_visualization(q1, cfg_radar, cfg_cfar):
    # GUI imports are deferred here so they only load in the main process
    # (child processes must not import GUI libraries)
    import warnings
    warnings.simplefilter("ignore", UserWarning)

    from scipy.interpolate import RegularGridInterpolator
    from direct.showbase.ShowBase import ShowBase
    from direct.task import Task

    import matplotlib
    matplotlib.use('Qt5Agg')  # use Qt5 backend for interactive display
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-v0_8-dark')

    from panda3d.core import loadPrcFileData
    loadPrcFileData('', 'window-type none')    # disable Panda3D native window
    loadPrcFileData('', 'audio-library-name null')  # disable audio

    from PyQt5 import QtWidgets
    from streaming_base.visualization.visualization import configure_ax_bf

    class MyApp(ShowBase):
        def __init__(self, queue_1, cfg_radar):
            ShowBase.__init__(self)

            self.q1 = queue_1
            self.latest_msg = {}   # stores latest frame per producer
            self.msg_count = set() # tracks which producers have sent a new frame

            # radar geometry parameters
            self.phi    = cfg_radar["phi"]         # azimuth angles (radians), 0→π
            self.r_idxs = cfg_radar["range_idx"]   # range bin indices

            # --- polar bird-eye-view plot ---
            self.fig_1 = plt.figure(figsize=(6, 6))
            self.ax_1 = self.fig_1.add_subplot(111, projection='polar')
            self.im = configure_ax_bf(self.ax_1, self.phi, self.r_idxs, 0, 0.3)

            # add meter labels to the radial axis instead of raw bin indices
            num_ticks   = 6
            radial_bins = np.linspace(self.r_idxs.min(), self.r_idxs.max(), num_ticks)
            radial_labels = [f"{r * RANGE_FACTOR:.2f}m" for r in radial_bins]
            self.ax_1.set_yticks(radial_bins)
            self.ax_1.set_yticklabels(radial_labels)
            self.ax_1.set_title("Bird Eye View (background removed)")

            # cartesian grid for the polar↔cartesian round-trip interpolation
            self.x = np.arange(-cfg_radar["width"], cfg_radar["width"], 1)
            self.y = self.r_idxs
            self.X, self.Y = np.meshgrid(self.x, self.y, indexing='xy')

            # running max used for frame-to-frame normalization
            # decays slowly so it adapts to changing signal levels
            self.running_max = 1.0

            # --- static clutter (background) removal ---
            self.clutter_frames = []   # accumulates frames during learning phase
            self.clutter_map    = None # average of learning frames = static background
            self.CLUTTER_LEARN  = 50   # number of frames to learn background (~2.5s at 20fps)

            # register the update function with Panda3D's task manager
            self.taskMgr.add(self.updateTask, "updateTask")

        def updateTask(self, task):
            # drain the queue, keeping only the latest frame
            try:
                while not self.q1.empty():
                    msg = self.q1.get_nowait()
                    if msg[0] == 'bev':
                        self.latest_msg[0] = msg[1]
                        self.msg_count.add(0)
            except:
                pass

            # only process when a new frame has arrived
            if self.msg_count == {0}:
                bf_1 = self.latest_msg[0]  # raw beamformed polar frame (phi x range)

                # --- polar → cartesian → polar round-trip ---
                # step 1: compute polar coordinates of each cartesian grid point
                phi1      = np.arctan2(self.Y.ravel(), self.X.ravel())
                r1        = np.hypot(self.X.ravel(), self.Y.ravel())
                cart2pol1 = np.column_stack((phi1, r1))

                # step 2: interpolate the beamformed frame onto the cartesian grid
                interp1 = RegularGridInterpolator(
                    (self.phi, self.r_idxs), bf_1,
                    method='linear', bounds_error=False, fill_value=0
                )
                Z_cart = interp1(cart2pol1).reshape(self.X.shape)

                # step 3: interpolate the cartesian map back to the original polar grid
                # this round-trip smooths interpolation artifacts
                interp_cart2pol = RegularGridInterpolator(
                    (self.y, self.x), Z_cart,
                    method='linear', bounds_error=False, fill_value=0
                )
                PHI, R = np.meshgrid(self.phi, self.r_idxs, indexing='ij')
                pts_back = np.column_stack((
                    (R * np.sin(PHI)).ravel(),  # x = r*sin(phi)
                    (R * np.cos(PHI)).ravel()   # y = r*cos(phi)
                ))
                Z_polar = interp_cart2pol(pts_back).reshape(PHI.shape)

                # flip axis 0 to match display orientation (0° at top)
                Z_polar = np.flip(Z_polar, axis=0)

                # --- normalization ---
                raw = np.abs(Z_polar)
                # update running max with slow decay so normalization stays stable
                self.running_max = max(self.running_max * 0.99, raw.max())
                to_plot = raw / self.running_max  # values now in [0, 1]

                # --- background learning phase (first CLUTTER_LEARN frames) ---
                if len(self.clutter_frames) < self.CLUTTER_LEARN:
                    self.clutter_frames.append(to_plot.copy())
                    # incrementally update clutter map as new frames arrive
                    self.clutter_map = np.mean(self.clutter_frames, axis=0)
                    remaining = self.CLUTTER_LEARN - len(self.clutter_frames)
                    self.ax_1.set_title(f"Learning background... ({remaining} frames left)")
                    # show raw (squared for contrast) during learning so display is not blank
                    self.im.set_array((to_plot ** 2).ravel())
                    self.fig_1.canvas.draw_idle()
                    QtWidgets.QApplication.processEvents()
                    self.msg_count.clear()
                    plt.pause(0.001)
                    return Task.cont  # skip detection until background is learned

                # --- background subtraction ---
                # subtract the learned static clutter map and clip negatives to 0
                # result: only dynamic (moving/changed) parts remain
                to_plot = np.clip(to_plot - self.clutter_map, 0, None)

                # square for display contrast: boosts strong returns, suppresses weak noise
                to_plot = to_plot ** 2

                # --- update display ---
                self.im.set_array(to_plot.ravel())
                self.fig_1.canvas.draw_idle()
                QtWidgets.QApplication.processEvents()
                self.msg_count.clear()
                plt.pause(0.001)

            return Task.cont  # tell Panda3D to call this task again next frame

    app = MyApp(q1, cfg_radar)
    app.run()


def main(cfg_radar, cfg_cfar):
    # create a queue with maxsize=1 so only the latest frame is kept
    # if the consumer is slow, old frames are dropped automatically
    q_main_1 = Queue(maxsize=1)

    # run the radar producer in a separate process so it doesn't block the GUI
    producer = Process(
        target=producer_real_time_1843,
        args=(q_main_1, cfg_radar, cfg_cfar, 4096, 4098, "192.168.33.30", "192.168.33.180"),
        daemon=True  # dies automatically when main process exits
    )
    producer.start()
    print("Producer started, launching visualization in main process...")

    # run visualization in the main process (required for GUI on most OSes)
    run_visualization(q_main_1, cfg_radar, cfg_cfar)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        producer.terminate()
        producer.join()
        print("Shutdown complete.")