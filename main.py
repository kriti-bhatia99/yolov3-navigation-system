# Importing the libraries
import cv2
import numpy as np
import gtts
from playsound import playsound


# Get video stream
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("http://192.168.43.8:81/stream")


# Setting the global variables
whT = 320
threshold = 0.8
nmsThreshold = 0.3


# Get the classes
classesFile = "coco.names"
classNames = []

with open(classesFile, 'rt') as file:
    classNames = file.read().rstrip("\n").split("\n")

objectQuantity = {classId:0 for classId in classNames}

# Get the model
modelConfiguration = "yolov3-320.cfg"
modelWeights = "yolov3-320.weights"

# modelConfiguration = "yolov3-tiny.cfg"
# modelWeights = "yolov3-tiny.weights"
# threshold = 0.7

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# Find the objects in the image
def findObjects(outputs, img):
    hT, wT, cT = img.shape
    # objects = []
    bbox = []
    classIds = []
    confidenceValues = []

    for key in objectQuantity.keys():
        objectQuantity[key] = 0

    # Get the detections
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence > threshold:
                w, h = int(detection[2] * wT), int(detection[3] * hT)
                x, y = int((detection[0] * wT) - w/2), int((detection[1] * hT) - h/2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confidenceValues.append(float(confidence))

    # Filtering out the overlapping detections
    indices = cv2.dnn.NMSBoxes(bbox, confidenceValues, threshold, nmsThreshold)

    for i in indices:
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        start_point = (x, y)
        end_point = (x + w, y + h)

        object = classNames[classIds[i]]
        objectQuantity[object] += 1
        print(objectQuantity)

        # Plotting the rectangle and the text
        cv2.rectangle(img, start_point, end_point, (255, 0, 255), 3)
        cv2.putText(img, f"{object.upper()} {int(confidenceValues[i]*100)}%",
        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


    # Save the audio output and playing it
    for o, qty in objectQuantity.items():
        if cv2.waitKey(1) == ord('q'):
            quit()

        if qty != 0:
            final_outcome = str(qty) + " " + o
            tts = gtts.gTTS(final_outcome)
            filename = "detections/" + final_outcome + ".mp3"
            tts.save(filename)
            playsound(filename)

# Start the camera
while True:
    success, img = cap.read()

    # Convert the read image to blob to feed it to the network
    blob = cv2.dnn.blobFromImage(img, 1/255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    # Get output layers
    layerNames = net.getLayerNames()
    outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]


    # Send the image and get the output of the output layers
    outputs = net.forward(outputNames)
    # print("1", outputs[0].shape) (300, 85)
    # print("2", outputs[1].shape) (1200, 85)
    # print("3", outputs[2].shape) (4800, 85)
    # 85 columns: center x, center y, width, height, confidence, probability of all 80 classes
    # 300 rows: 300 outputs
    findObjects(outputs, img)

    # Plot the detected objects and display the frame
    cv2.imshow("Image", img)

    # Close the camera
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
