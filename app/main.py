from math import hypot
import dlib
import numpy as np
from flask import Flask, Response, render_template
import cv2

app = Flask(__name__, template_folder='templates')
video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier()
face_cascade.load(cv2.samples.findFile("../venv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml"))


def genDog(video, imgName):

    # Loading Camera and Nose image and Creating mask
    nose_image = cv2.imread(imgName)
    boo, frame = video.read()
    rows, cols, boo = frame.shape
    nose_mask = np.zeros((rows, cols), np.uint8)

    # Loading Face detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    while True:
        boo, frame = video.read()
        nose_mask.fill(0)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(frame)

        for face in faces:
            landmarks = predictor(gray_frame, face)

            # Nose coordinates
            top_nose = (landmarks.part(29).x, landmarks.part(29).y)
            center_nose = (landmarks.part(30).x, landmarks.part(30).y)
            left_nose = (landmarks.part(31).x, landmarks.part(31).y)
            right_nose = (landmarks.part(35).x, landmarks.part(35).y)

            nose_width = int(hypot(left_nose[0] - right_nose[0] * 1.5,
                                   left_nose[1] - right_nose[1]))
            nose_height = int(nose_width * 0.77)

            # New nose position
            top_left = (int(center_nose[0] - nose_width / 2),
                        int(center_nose[1] - nose_height / 2))
            bottom_right = (int(center_nose[0] + nose_width / 2),
                            int(center_nose[1] + nose_height / 2))

            # Adding the new nose
            nose_img = cv2.resize(nose_image, (nose_width, nose_height))
            nose_img_gray = cv2.cvtColor(nose_img, cv2.COLOR_BGR2GRAY)
            boo, nose_mask = cv2.threshold(nose_img_gray, 25, 255, cv2.THRESH_BINARY_INV)

            nose_area = frame[top_left[1]: top_left[1] + nose_height,
                        top_left[0]: top_left[0] + nose_width]
            nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)
            final_nose = cv2.add(nose_area_no_nose, nose_img)

            frame[top_left[1]: top_left[1] + nose_height,
            top_left[0]: top_left[0] + nose_width] = final_nose

        ret, jpeg = cv2.imencode('.jpg', frame)

        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def genClown(video, imgName):

    # Loading Camera and Nose image and Creating mask
    nose_image = cv2.imread(imgName)
    boo, frame = video.read()
    rows, cols, boo = frame.shape
    nose_mask = np.zeros((rows, cols), np.uint8)

    # Loading Face detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    while True:
        boo, frame = video.read()
        nose_mask.fill(0)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(frame)

        for face in faces:
            landmarks = predictor(gray_frame, face)

            # Nose coordinates
            top_nose = (landmarks.part(29).x, landmarks.part(29).y)
            center_nose = (landmarks.part(30).x, landmarks.part(30).y)
            left_nose = (landmarks.part(31).x, landmarks.part(31).y)
            right_nose = (landmarks.part(35).x, landmarks.part(35).y)

            nose_width = int(hypot(left_nose[0] - right_nose[0] * 1.2,
                                   left_nose[1] - right_nose[1]))
            nose_height = int(nose_width * 0.77)

            # New nose position
            top_left = (int(center_nose[0] - nose_width / 2),
                        int(center_nose[1] - nose_height / 2))
            bottom_right = (int(center_nose[0] + nose_width / 2),
                            int(center_nose[1] + nose_height / 2))

            # Adding the new nose
            nose_img = cv2.resize(nose_image, (nose_width, nose_height))
            nose_img_gray = cv2.cvtColor(nose_img, cv2.COLOR_BGR2GRAY)
            boo, nose_mask = cv2.threshold(nose_img_gray, 25, 255, cv2.THRESH_BINARY_INV)

            nose_area = frame[top_left[1]: top_left[1] + nose_height,
                        top_left[0]: top_left[0] + nose_width]
            nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)
            final_nose = cv2.add(nose_area_no_nose, nose_img)

            frame[top_left[1]: top_left[1] + nose_height,
            top_left[0]: top_left[0] + nose_width] = final_nose

        ret, jpeg = cv2.imencode('.jpg', frame)

        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/dog_nose')
def dog_nose():
    # Set to global because we refer the video variable on global scope,
    # Or in other words outside the function
    global video

    # Return the result on the web
    return Response(genDog(video, imgName="images/dognose.png"), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/clown_nose')
def clown_nose():
    # Set to global because we refer the video variable on global scope,
    # Or in other words outside the function
    global video

    # Return the result on the web
    return Response(genClown(video, imgName="images/clownnose.png"), mimetype='multipart/x-mixed-replace; '
                                                                              'boundary=frame')


@app.route('/')
def index():
    return render_template("index.html")


if __name__ == '__main__':
    app.run(port=5000, threaded=True)
