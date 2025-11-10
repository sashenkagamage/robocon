import torch
from ultralytics import YOLO
from multiprocessing import freeze_support

if __name__ == "__main__":
    freeze_support()

    print("")
    print("")
    print("==========================================")
    print("IC_AI_Model_01 Training Start")
    print("==========================================")
    print("")       
    print("")

    m_IC_AI_Model = YOLO("IC_AI_Model_00/yolo11n.pt")

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        m_Device = torch.device("cuda")
    else:
        print("Using CPU")
        m_Device = torch.device("cpu")

    m_IC_AI_Model_Train_Result = m_IC_AI_Model.train(data="IC_AI_Model_01/dataset.yaml",
                                                     epochs  = 100,
                                                     imgsz   = 640,
                                                     project = 'IC_AI_Model_01/result',
                                                     device  = m_Device)
    m_IC_AI_Model_Export_Result  = m_IC_AI_Model.export()

    print("")
    print("")
    print("==========================================")
    print("IC_AI_Model_01 Training Finish")
    print("==========================================")
    print("")
    print("")