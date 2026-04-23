import os
from multiprocessing import Process, Queue
import time

# On importe les fonctions principales de tes futurs 3 fichiers
# (Il faudra que ces fonctions existent dans tes fichiers respectifs)
from streaming_base.streaming.new.producer import producer_task
from streaming_base.streaming.new.processor import processor_task
from streaming_base.streaming.new.visualizer_old import run_visualization

def main(cfg_radar, cfg_cfar):
    # ---------------------------------------------------------
    # 1. CONFIGURATION
    # ---------------------------------------------------------
    # Ici tu mettras tes dictionnaires cfg_radar et cfg_cfar
    # Pour l'instant on les imagine récupérés via tes scripts existants
    #cfg_radar = { "samples_per_chirp": 256, "num_tx": 3, "num_rx": 4 } 
    #cfg_cfar = { "cfar_on": True }

    # ---------------------------------------------------------
    # 2. CRÉATION DES QUEUES (LES TUYAUX)
    # ---------------------------------------------------------
    # q_raw : Transporte les données brutes ADC (Producer -> Processor)
    q_raw = Queue(maxsize=1) 
    
    # q_results : Transporte la Heatmap + les Tracks (Processor -> Visualizer)
    q_results = Queue(maxsize=1)

    print("--- Initialisation du système Radar Multi-Cœurs ---")

    # ---------------------------------------------------------
    # 3. DÉFINITION DES PROCESSUS (LES CŒURS)
    # ---------------------------------------------------------
    
    # Cœur 1 : Acquisition (Producer)
    p_prod = Process(
        target=producer_task, 
        args=(q_raw, cfg_radar),
        name="Radar_Producer",
        daemon=True
    )

    # Cœur 2 : Calculs & Tracking (Processor)
    p_proc = Process(
        target=processor_task, 
        args=(q_raw, q_results, cfg_radar, cfg_cfar),
        name="Radar_Processor",
        daemon=True
    )

    # ---------------------------------------------------------
    # 4. LANCEMENT
    # ---------------------------------------------------------
    
    p_prod.start()
    print("[OK] Cœur 1 : Producer démarré.")
    
    p_proc.start()
    print("[OK] Cœur 2 : Processor démarré.")

    # Cœur 3 : Visualisation (Visualizer)
    # On le lance dans le processus principal (Main) car les interfaces
    # graphiques (GUI) n'aiment pas être dans des processus "enfants".
    print("[OK] Cœur 3 : Lancement du Visualizer...")
    try:
        run_visualization(q_results, cfg_radar)
    except KeyboardInterrupt:
        print("\nArrêt du système demandé...")
    finally:
        # Nettoyage des processus pour ne pas laisser de "zombies"
        p_prod.terminate()
        p_proc.terminate()
        p_prod.join()
        p_proc.join()
        print("Système arrêté proprement.")

if __name__ == "__main__":
    main()