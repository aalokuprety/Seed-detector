import yaml
from ultralytics import YOLO
import os
import shutil

class SeedCounterTrainer:
    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.data_yaml_path = os.path.join(project_dir, 'data', 'dataset.yaml')
        
    def create_data_yaml(self):
        """Create the dataset.yaml file required by YOLOv8"""
        data_yaml = {
            'path': os.path.join(self.project_dir, 'data'),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 1,  # number of classes
            'names': ['seed']  # class names
        }
        
        with open(self.data_yaml_path, 'w') as f:
            yaml.dump(data_yaml, f)
        
        print("dataset.yaml created successfully!")
    
    def train_model(self, epochs=100):
        """Train the YOLOv8 model"""
        # Load a pre-trained YOLOv8 model
        model = YOLO('yolov8n.yaml')  # 'n' for nano, can use 's' for small or 'm' for medium
        
        # Train the model
        results = model.train(
            data=self.data_yaml_path,
            epochs=epochs,
            imgsz=640,
            batch=16,
            name='seed_detector'
        )
        
        print("Training completed!")
        return results

# Example usage
if __name__ == "__main__":
    trainer = SeedCounterTrainer("seed_counter_project")
    trainer.create_data_yaml()
    results = trainer.train_model(epochs=100)
