import albumentations as A
import cv2
import os
import numpy as np
from tqdm import tqdm
import shutil

class DataAugmenter:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Create strong augmentation pipeline
        self.transform_strong = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ], p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1),
                A.GaussianBlur(blur_limit=(3, 7), p=1),
            ], p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=45,
                p=0.5
            ),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))
        
        # Create mild augmentation pipeline
        self.transform_mild = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.5
            ),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))

    def read_yolo_annotations(self, annotation_path):
        """Read YOLO format annotations"""
        bboxes = []
        class_labels = []
        
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    bboxes.append([x_center, y_center, width, height])
                    class_labels.append(class_id)
                    
        return np.array(bboxes), class_labels

    def save_yolo_annotations(self, bboxes, class_labels, annotation_path):
        """Save annotations in YOLO format"""
        with open(annotation_path, 'w') as f:
            for bbox, class_id in zip(bboxes, class_labels):
                f.write(f"{int(class_id)} {' '.join(map(str, bbox))}\n")

    def augment_dataset(self, augmentations_per_image=3):
        """Augment entire dataset"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Get list of all images
        image_files = [f for f in os.listdir(os.path.join(self.input_dir, 'images'))
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Found {len(image_files)} images. Generating {augmentations_per_image} augmentations each...")
        
        for img_file in tqdm(image_files):
            # Read image
            img_path = os.path.join(self.input_dir, 'images', img_file)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Read annotations
            ann_file = os.path.splitext(img_file)[0] + '.txt'
            ann_path = os.path.join(self.input_dir, 'labels', ann_file)
            bboxes, class_labels = self.read_yolo_annotations(ann_path)
            
            # Copy original files
            shutil.copy2(img_path, os.path.join(self.output_dir, 'images', img_file))
            if os.path.exists(ann_path):
                shutil.copy2(ann_path, os.path.join(self.output_dir, 'labels', ann_file))
            
            # Generate augmentations
            for i in range(augmentations_per_image):
                # Alternate between strong and mild augmentations
                transform = self.transform_strong if i % 2 == 0 else self.transform_mild
                
                # Apply augmentation
                transformed = transform(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
                
                # Save augmented image
                aug_img_name = f"{os.path.splitext(img_file)[0]}_aug{i}{os.path.splitext(img_file)[1]}"
                aug_img_path = os.path.join(self.output_dir, 'images', aug_img_name)
                aug_ann_path = os.path.join(self.output_dir, 'labels', 
                                          f"{os.path.splitext(aug_img_name)[0]}.txt")
                
                # Save image
                aug_image = cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)
                cv2.imwrite(aug_img_path, aug_image)
                
                # Save annotations
                self.save_yolo_annotations(
                    transformed['bboxes'],
                    transformed['class_labels'],
                    aug_ann_path
                )

def augment_seed_dataset(input_dir, output_dir, augmentations_per_image=3):
    """Helper function to run augmentation"""
    # Create output directories if they don't exist
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    
    # Initialize and run augmentation
    augmenter = DataAugmenter(input_dir, output_dir)
    augmenter.augment_dataset(augmentations_per_image)

# Example usage
if __name__ == "__main__":
    augment_seed_dataset(
        input_dir='seed_counter_project/data/original',
        output_dir='seed_counter_project/data/augmented',
        augmentations_per_image=3
    )