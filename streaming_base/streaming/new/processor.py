import numpy as np
import time
import queue
from streaming_base.processing.processing import process_frame_2d, beamform_2d
from streaming_base.utils.utils import get_ant_pos_2d

def processor_task(q_raw, q_results, cfg_radar, cfg_cfar):
    """
    Cœur 2 : Le Cerveau.
    Transforme les données ADC brutes en Heatmap et liste d'objets (Tracks).
    """
    # --- INITIALISATION ---
    r_idxs = cfg_radar["range_idx"]
    num_tx = cfg_radar["num_tx"]
    num_rx = cfg_radar["num_rx"]
    chirp_loops = cfg_radar["num_doppler"]
    adc_samples = cfg_radar["samples_per_chirp"]
    
    # Mémoire pour le lissage/moyennage (comme dans ton ancien code)
    last_frames = np.zeros((5, num_tx * num_rx, chirp_loops, adc_samples), dtype=np.complex64)
    x_locs, _, _ = get_ant_pos_2d(num_tx * num_rx, adc_samples, num_rx)
    
    print("[PROCESSOR] Prêt à traiter les données...")

    while True:
        try:
            # 1. Récupération de la frame brute depuis le Producer
            # On attend que le Producer envoie quelque chose (bloquant)
            raw = q_raw.get() 

            # 2. PRÉ-TRAITEMENT (Hamming + Reshape)
            adc_windowed = raw * np.hamming(adc_samples)
            adc_windowed = adc_windowed.reshape(chirp_loops, num_tx, num_rx, adc_samples)
            adc_windowed = adc_windowed.transpose(1, 2, 0, 3) # tx, rx, loops, samples
            adc_windowed = adc_windowed.reshape(num_tx * num_rx, chirp_loops, adc_samples)

            # 3. FFT RANGE
            range_fft = np.fft.fft(adc_windowed, axis=-1)
            range_fft_s = range_fft[:, :, r_idxs]
            range_fft_s[:, :, 0:4] = 0  # Suppression du bruit de proximité

            # 4. ACCUMULATION (Moyenne glissante)
            last_frames[:-1] = last_frames[1:]
            last_frames[-1] = range_fft_s
            bf_input = np.mean(last_frames, axis=0)

            # 5. BEAMFORMING (Calcul de la Heatmap)
            # Cette étape est la plus gourmande en CPU
            bf_output = beamform_2d(bf_input.squeeze(), cfg_radar, x_locs[:, 0])
            
            # 6. DÉTECTION (CFAR)
            max_output = abs(bf_output).max() if abs(bf_output).max() != 0 else 1.0
            if cfg_cfar['cfar_on']:
                heatmap = process_frame_2d(abs(bf_output)**2, cfg_cfar) / max_output
            else:
                heatmap = abs(bf_output) / max_output

            # ---------------------------------------------------------
            # 7. ZONE MULTI-TRACKING (À CODER)
            # ---------------------------------------------------------
            # Ici, tu analyseras 'heatmap' pour extraire des coordonnées (x, y)
            # Pour l'instant, on envoie une liste vide de tracks.
            tracks = [] 
            
            # Exemple de structure future :
            # detections = cluster_points(heatmap)
            # tracks = kalman_filter_update(detections)
            # ---------------------------------------------------------

            # 8. ENVOI AU VISUALIZER (Cœur 3)
            try:
                # On envoie un dictionnaire complet
                payload = {
                    "heatmap": heatmap,
                    "tracks": tracks,  # Tes futurs IDs et positions
                    "timestamp": time.time()
                }
                q_results.put_nowait(payload)
            except queue.Full:
                continue

        except Exception as e:
            print(f"[PROCESSOR] Erreur : {e}")
            continue