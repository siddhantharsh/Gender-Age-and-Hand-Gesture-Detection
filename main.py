import threading
import cv2
from app import main as hand_gesture_detection_main  # Import the main function from app.py
from RealtimeGenderClassificationWebcam import getFace as gender_classification_main

def webcam_feed():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run gender classification on the frame
        gender_classification_frame = gender_classification_main(frame)
        
        # Display the result of gender classification
        cv2.imshow("Gender Classification", gender_classification_frame)
        
        # Run hand gesture detection in parallel (by calling its main function)
        hand_gesture_detection_main()  # This will internally handle its own webcam feed
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    webcam_thread = threading.Thread(target=webcam_feed)
    webcam_thread.start()
    webcam_thread.join()
