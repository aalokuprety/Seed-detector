# 1. Create a new conda environment (recommended)
# Run these commands in your terminal:
#conda create -n seed_counter python=3.10
#conda activate seed_counter

# 2. Install required packages
#pip install ultralytics  # for YOLOv8
#pip install opencv-python
#pip install albumentations
#pip install numpy
#pip install pillow

# 3. Create project structure
import os

def create_project_structure():
    # Create main project directory
    base_dir = "seed_counter_project"
    directories = [
        "data/images/train",
        "data/images/val",
        "data/images/test",
        "data/labels/train",
        "data/labels/val",
        "data/labels/test",
        "models",
        "results"
    ]
    
    for dir_path in directories:
        os.makedirs(os.path.join(base_dir, dir_path), exist_ok=True)
    
    print("Project structure created successfully!")

# Run the function to create directories
create_project_structure()