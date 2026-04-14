capture_file               =   "task4_student"

--TODO: edit this path!!
SAVE_DATA_PATH = "C:\\\\Users\\\\mathy\\\\Dev\\\\com-405-radar-tutorial\\\\data\\\\" .. capture_file .. ".bin"

ar1.CaptureCardConfig_StartRecord(SAVE_DATA_PATH, 1)
ar1.StartFrame()

