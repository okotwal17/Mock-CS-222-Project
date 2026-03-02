import cv2
import time
from ultralytics import YOLO

def main():
    # --- FPS SETTINGS ---
    TARGET_FPS = 1  # Set your desired FPS here
    delay = 1 / TARGET_FPS
    # --------------------

    # Load a pretrained YOLOv8n model
    model = YOLO('yolov8n.pt')

    # Open the webcam
    cap = cv2.VideoCapture(0)

    # Optional: Try to set the camera hardware FPS (not supported by all webcams)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print(f"Phone detection started at target {TARGET_FPS} FPS. Press 'q' to quit.")

    prev_time = 0
    
    while True:
        start_time = time.time()
        
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Run inference (stream=True is more efficient for video)
        results = model(frame, verbose=False)

        phone_detected = False

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Class index 67 is 'cell phone'
                if int(box.cls) == 67:
                    phone_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame, "Phone", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display detection status
        if phone_detected:
            status_text = "phone detected"
            color = (0, 255, 0)
        else:
            status_text = "no phone detected"
            color = (0, 0, 255)

        cv2.putText(frame, status_text, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # --- CALCULATE AND DISPLAY ACTUAL FPS ---
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        
        cv2.putText(frame, f"FPS: {int(fps)}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # ----------------------------------------

        # Show frame
        cv2.imshow('Phone Detector', frame)

        # --- FPS THROTTLING ---
        # Calculate how much time we spent and sleep if we are too fast
        elapsed = time.time() - start_time
        sleep_time = max(1, int((delay - elapsed) * 1000)) if delay > elapsed else 1
        
        if cv2.waitKey(sleep_time) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
