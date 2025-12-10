import torch
import cv2
from yolov5.utils.torch_utils import select_device

# Load YOLOv5 model
model_path = "yolov5s.pt"
device = select_device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model properly with AutoShape enabled
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
model.to(device).eval()

# Object detection function
def detect_weapon(frame):
    results = model(frame)  # AutoShape handles preprocessing
    results.render()  # Draw bounding boxes on frame
    return frame

# Real-time detection function
def run_detection(video_source=0):
    cap = cv2.VideoCapture(video_source)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = detect_weapon(frame)
        cv2.imshow("Weapon Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Run real-time detection
if __name__ == "__main__":
    run_detection()
