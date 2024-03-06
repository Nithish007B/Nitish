import cv2
import numpy as np
import time
import torch
from collections import deque, defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort 

# Function to estimate speed based on the tracking logic from the first code snippet
def estimateSpeed(location1, location2, fps, pixel_to_meter_ratio):
    dx = location2[0] - location1[0]
    dy = location2[1] - location1[1]
    distance_pixels = np.sqrt(dx**2 + dy**2)
    distance_meters = distance_pixels * pixel_to_meter_ratio
    speed_mps = distance_meters * fps
    speed_kmh = speed_mps * 3.6
    return speed_kmh

class YoloDetector():
    def __init__(self, model_name):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.load_model(model_name).to(self.device)
        self.classes = self.model.names
        print("Using Device: ", self.device)

    def load_model(self, model_name):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True) if model_name else torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        return self.classes[int(x)]

    def plot_boxes(self, results, frame, height, width, confidence=0.3):
        labels, cord = results
        detections = []
        for i in range(len(labels)):
            row = cord[i]
            if row[4] >= confidence:
                x1, y1, x2, y2 = int(row[0] * width), int(row[1] * height), int(row[2] * width), int(row[3] * height)
                if self.class_to_label(labels[i]) == 'car':
                    detections.append(([x1, y1, x2 - x1, y2 - y1], row[4].item(), 'car'))
        return frame, detections

detector = YoloDetector(model_name=None)

object_tracker = DeepSort(
    max_age=5,
    n_init=1,
    nms_max_overlap=1.0,
    max_cosine_distance=0.3,
    nn_budget=None,
    override_track_class=None,
    embedder="mobilenet",
    half=True,
    bgr=True,
    embedder_gpu=True,
    embedder_model_name=None,
    embedder_wts=None,
    polygon=False,
    today=None)

cap = cv2.VideoCapture('D:/Y5DS/tihan1.webm')
fps = cap.get(cv2.CAP_PROP_FPS)
pixel_to_meter_ratio = 0.05  # Calibrate this value
speed_history_length = 5  
speed_history = defaultdict(lambda: deque(maxlen=speed_history_length), {})

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()
    
    results = detector.score_frame(frame)
    frame, detections = detector.plot_boxes(results, frame, frame.shape[0], frame.shape[1], confidence=0.5)
    tracks = object_tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        
        track_id = track.track_id
        bbox = track.to_tlbr()  # to get bounding box in (top, left, bottom, right) format
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

        speed_kmh = 0.0  # Default speed
        if len(speed_history[track_id]) >= 2:
            # Calculate speed using the last and current positions
            location1 = speed_history[track_id][-2]
            location2 = speed_history[track_id][-1]
            speed_kmh = estimateSpeed(location1, location2, fps, pixel_to_meter_ratio)

        speed_history[track_id].append(center)  # Update speed history with the current center

        cv2.putText(frame, f"ID: {track_id} Speed: {speed_kmh:.2f} km/h", 
                    (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 0), 2)

        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                      (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    elapsed_time = time.time() - start_time
    if elapsed_time < 1/fps:
        time.sleep(1/fps - elapsed_time)

cap.release()
cv2.destroyAllWindows()
