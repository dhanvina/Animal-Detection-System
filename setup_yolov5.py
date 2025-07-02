import os
import torch
from pathlib import Path

def setup_yolov5():
    print("Setting up YOLOv5...")
    
    # Clone YOLOv5 repository if not exists
    if not os.path.exists('yolov5'):
        print("Cloning YOLOv5 repository...")
        os.system('git clone https://github.com/ultralytics/yolov5')
    
    # Install YOLOv5 requirements
    print("Installing YOLOv5 requirements...")
    os.system('pip install -r yolov5/requirements.txt')
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('uploads', exist_ok=True)
    
    print("\nYOLOv5 setup complete!")
    print("You can now use YOLOv5 for animal detection.")
    print("The pre-trained model will be automatically downloaded when you first run the detection.")

if __name__ == "__main__":
    setup_yolov5()
