import numpy as np
import queue
from streaming_base.mmwave.dataloader.adcv3 import DCA1000

def producer_task(q_raw, cfg_radar, static_ip="192.168.33.30", system_ip="192.168.33.180"):
    """
    Cœur 1 : Acquisition de données brutes.
    Lit le flux UDP du DCA1000 et transmet la matrice ADC organisée.
    """
    # Extraction des paramètres nécessaires
    num_tx = cfg_radar["num_tx"]
    num_rx = cfg_radar["num_rx"]
    chirp_loops = cfg_radar["num_doppler"]
    adc_samples = cfg_radar["samples_per_chirp"]

    print(f"[PRODUCER] Initialisation DCA1000 (IP: {static_ip})")
    
    dca = DCA1000()
    # Configuration du capteur via la DLL/Driver
    dca.sensor_config(
        chirps=num_tx, 
        chirp_loops=chirp_loops, 
        num_rx=num_rx, 
        num_samples=adc_samples
    )
    
    print("[PRODUCER] DCA1000 prêt. Lecture du flux...")

    try:
        while True:
            # 1. Lecture des paquets bruts sur le port data
            adc_data = dca.read()
            
            # 2. Organisation des données (Récupération de la structure spatiale)
            # Retourne une forme : [frames x chirps x samples x rx]
            raw_frame = dca.organize(
                raw_frame=adc_data, 
                num_chirps=num_tx * chirp_loops,
                num_rx=num_rx, 
                num_samples=adc_samples, 
                num_frames=1, 
                model='1843'
            )

            if raw_frame is None:
                continue

            # 3. Envoi au Processor (Cœur 2)
            # On utilise put_nowait pour ne jamais bloquer la lecture UDP
            try:
                # On envoie la frame brute "propre"
                q_raw.put_nowait(raw_frame)
            except queue.Full:
                # Si le processeur est trop lent, on jette la frame 
                # pour rester synchronisé avec le temps réel
                continue

    except KeyboardInterrupt:
        print("[PRODUCER] Arrêt par l'utilisateur.")
    except Exception as e:
        print(f"[PRODUCER] Erreur critique : {e}")