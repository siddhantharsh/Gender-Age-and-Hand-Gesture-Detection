import cv2

def getFace(faceDetectionModel, inputImage, conf_threshold=0.7):
    cpy_input_image = inputImage.copy()  # To avoid modifications to the original input

    frameWidth = cpy_input_image.shape[1]
    frameHeight = cpy_input_image.shape[0]

    blob = cv2.dnn.blobFromImage(cpy_input_image, scalefactor=1, size=(227, 227), mean=(104, 117, 123), crop=False)  # preprocessed image

    faceDetectionModel.setInput(blob)
    detections = faceDetectionModel.forward()

    bounding_boxes = []
    for i in range(detections.shape[2]):  # detections is an array having [no.of.images/batch size, classes/channels, i-th detections, confidence_score]
        confidence_score = detections[0, 0, i, 2]  # gets the confidence score for i-detections, 4-th index(value:2) shows confidence score

        if confidence_score > conf_threshold:  # get the co-ordinates of the bounding boxes only if it's detected as a face, confidence score sets the minimum limit
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bounding_boxes.append([x1, y1, x2, y2])

            cv2.rectangle(cpy_input_image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
    return cpy_input_image, bounding_boxes

# loading the face detection model
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# loading the gender detection pretrained model
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
genderNet = cv2.dnn.readNet(genderModel, genderProto)

genderList = ['Male', 'Female']

# realtime
cap = cv2.VideoCapture(0)

while True:
    ret, inputImage = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    detected_image, bounding_boxes = getFace(faceNet, inputImage)

    male_count = 0
    female_count = 0

    if not bounding_boxes:
        print("No faces detected in the image")
    else:
        for bounding_box in bounding_boxes:
            x1, y1, x2, y2 = bounding_box

            # Ensure bounding box coordinates are within the image dimensions
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(inputImage.shape[1], x2)
            y2 = min(inputImage.shape[0], y2)

            # Check if the bounding box has valid dimensions
            if x2 > x1 and y2 > y1:
                detected_face_box = inputImage[y1:y2, x1:x2]

                # Ensure the detected face box is not empty
                if detected_face_box.size > 0:
                    detected_face_blob = cv2.dnn.blobFromImage(detected_face_box, scalefactor=1, size=(227, 227), mean=([78.4263377603, 87.7689143744, 114.895847746]), crop=False)

                    genderNet.setInput(detected_face_blob)
                    genderPrediction = genderNet.forward()

                    gender = genderList[genderPrediction[0].argmax()]

                    if gender == 'Male':
                        male_count += 1
                    else:
                        female_count += 1

                    cv2.putText(img=detected_image, text=gender, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1.1, color=(0, 255, 0), thickness=2)
        
        # Display gender count on the frame
        cv2.putText(detected_image, f'Male: {male_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(detected_image, f'Female: {female_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Show a warning if one female and more than one male are detected
        if female_count == 1 and male_count > 1:
            cv2.putText(detected_image, "Warning: 1 Female and more than 1 Male detected", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
    cv2.imshow("Detected Image", detected_image)
       
    if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
        break
    
cap.release()
cv2.destroyAllWindows()
