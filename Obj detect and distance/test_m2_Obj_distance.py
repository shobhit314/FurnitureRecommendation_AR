import cv2
import numpy as np
from ultralytics import YOLO

# --- 1. Model Loading ---

def load_model(model_path='yolov8n.pt'):
    """Loads the YOLOv8 model from the specified path."""
    model = YOLO(model_path)
    return model

# --- 2. Scene Detection Function ---

def detect_scene(frame, model):
    """
    Detects objects in a given camera frame using the YOLO model.
    Returns a list of detected objects with their details.
    """
    results = model(frame, stream=True, verbose=False)
    detected_objects = []
    
    for result in results:
        names = result.names
        for box in result.boxes:
            if box.conf > 0.6:  # Confidence threshold
                detected_objects.append({
                    "label": names[int(box.cls)],
                    "confidence": float(box.conf),
                    "box": box.xyxy[0].cpu().numpy().astype(int) # [x1, y1, x2, y2]
                })
    return detected_objects

# --- 3. Optimal Placement Calculation Function ---

def calculate_optimal_placement(detected_objects, new_object_dims):
    """
    Calculates the best placement spot on a detected surface.
    For this simulation, we'll find empty space on the largest detected 'dining table'.
    """
    # Find the largest dining table to use as our target surface
    target_surface = None
    max_area = 0
    for obj in detected_objects:
        if obj["label"] == 'dining table':
            box = obj["box"]
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area > max_area:
                max_area = area
                target_surface = obj
    
    if not target_surface:
        return None # No suitable surface found

    # Define a grid of candidate points on the surface
    surface_box = target_surface["box"]
    step = 20
    candidate_points = []
    
    x_coords = np.arange(surface_box[0] + step, surface_box[2] - step, step)
    y_coords = np.arange(surface_box[1] + step, surface_box[3] - step, step)
    
    if len(x_coords) == 0 or len(y_coords) == 0:
        return None # Surface is too small

    for x in x_coords:
        for y in y_coords:
            candidate_points.append(np.array([x, y]))

    if not candidate_points:
        return None

    # Score each point based on distance from other objects
    best_point = None
    max_score = -1

    for point in candidate_points:
        # Calculate the minimum distance to any other object on the table
        min_dist = float('inf')
        is_colliding = False
        
        for obj in detected_objects:
            if obj is target_surface:
                continue # Don't check against the table itself

            obj_box = obj["box"]
            # Check if the point is inside another object's box (collision)
            if obj_box[0] < point[0] < obj_box[2] and obj_box[1] < point[1] < obj_box[3]:
                is_colliding = True
                break
            
            # Simple distance to the center of the object
            obj_center = np.array([(obj_box[0] + obj_box[2]) / 2, (obj_box[1] + obj_box[3]) / 2])
            dist = np.linalg.norm(point - obj_center)
            if dist < min_dist:
                min_dist = dist
        
        if is_colliding:
            continue # Skip points that are inside an existing object

        # The score is the distance to the nearest object. We want to maximize this.
        score = min_dist
        
        if score > max_score:
            max_score = score
            best_point = point
            
    return best_point.astype(int) if best_point is not None else None

def calculate_object_distances(detected_objects):
    """
    Calculates distances between all detected objects in the frame.
    Returns a list of tuples: (label1, label2, distance_in_pixels).
    """
    distances = []
    centers = []

    # Step 1: get centers of bounding boxes
    for obj in detected_objects:
        box = obj["box"]
        center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
        centers.append((obj["label"], center))

    # Step 2: calculate pairwise distances
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            label1, c1 = centers[i]
            label2, c2 = centers[j]
            dist = np.linalg.norm(np.array(c1) - np.array(c2))
            distances.append((label1, label2, dist))

    return distances


# --- 4. Visualization Function ---

def draw_visualizations(frame, detected_objects, best_placement, distances):
    """Draws bounding boxes, optimal placement, and object-to-object distances."""
    # Draw bounding boxes
    for obj in detected_objects:
        box = obj["box"]
        label = f"{obj['label']} {obj['confidence']:.2f}"
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
        cv2.putText(frame, label, (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Draw placement
    if best_placement is not None:
        cv2.circle(frame, tuple(best_placement), 15, (0, 255, 0), -1)
        cv2.putText(frame, "Place Here", (best_placement[0] - 40, best_placement[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw distances between objects
    for (label1, label2, dist) in distances:
        text = f"{label1}-{label2}: {int(dist)}px"
        cv2.putText(frame, text, (30, 30 + 20 * distances.index((label1, label2, dist))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return frame


# --- Main Execution Loop (MODIFIED FOR SMARTPHONE) ---

if __name__ == "__main__":
    print("Loading model...")
    model = load_model()
    
    # === CHANGE IS HERE ===
    # Replace the IP address with the one shown on your phone's app
    url = "http://10.9.6.132:8080/video" 
    print(f"Connecting to smartphone camera at: {url}")
    
    cap = cv2.VideoCapture(url) # Use the URL instead of 0
    # ======================

    if not cap.isOpened():
        print("Error: Cannot open camera stream. Make sure the IP address is correct and your phone is on the same Wi-Fi.")
        exit()

    new_object_dimensions = (50, 50) 

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame from stream.")
            break

        detected_objects = detect_scene(frame, model)
        optimal_spot = calculate_optimal_placement(detected_objects, new_object_dimensions)
        distances = calculate_object_distances(detected_objects)

        output_frame = draw_visualizations(frame, detected_objects, optimal_spot, distances)


        cv2.imshow("AR Placement Logic - Smartphone Feed", output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Stream ended.")