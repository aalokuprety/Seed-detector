import cv2
from ultralytics import YOLO

class SeedCounter:
    def __init__(self, model_path):
        """
        Initialize the seed counter with a trained model
        model_path: path to your trained model weights (e.g., 'best.pt')
        """
        self.model = YOLO(model_path)
    
    def count_seeds(self, image_path, confidence=0.25):
        """
        Count seeds in an image
        image_path: path to the image file
        confidence: detection confidence threshold
        """
        # Run inference
        results = self.model.predict(
            source=image_path,
            conf=confidence,
            save=True  # Save annotated image
        )
        
        # Get number of detected seeds
        num_seeds = len(results[0].boxes)
        
        # Get confidence scores
        confidence_scores = results[0].boxes.conf.cpu().numpy()
        
        return {
            'count': num_seeds,
            'confidence_scores': confidence_scores,
            'results': results[0]
        }
    
    def visualize_detections(self, image_path, results):
        """
        Visualize detected seeds on the image
        """
        # Read image
        image = cv2.imread(image_path)
        
        # Draw boxes
        for box in results.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add count
        cv2.putText(
            image, 
            f'Seeds: {len(results.boxes)}', 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
        
        return image

# Example usage
if __name__ == "__main__":
    # Initialize counter with your trained model
    counter = SeedCounter('path/to/your/best.pt')
    
    # Count seeds in an image
    image_path = 'path/to/test/image.jpg'
    results = counter.count_seeds(image_path)
    
    print(f"Number of seeds detected: {results['count']}")
    
    # Visualize results
    annotated_image = counter.visualize_detections(image_path, results['results'])
    cv2.imwrite('annotated_image.jpg', annotated_image)
