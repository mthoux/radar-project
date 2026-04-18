capture_file               =   "v"

--TODO: edit this path!!
SAVE_DATA_PATH = "C:\\\\Users\\\\USER\\\\OneDrive\\\\Escritorio\\\\EPFL\\\\IntelligentSystems-Comm\\\\radar-project\\\\data\\\\" .. capture_file .. ".bin"

ar1.CaptureCardConfig_StartRecord(SAVE_DATA_PATH, 1)
ar1.StartFrame()

