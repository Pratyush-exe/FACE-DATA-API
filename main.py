from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import base64
import cv2
import numpy as np
from keras.models import model_from_json

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config['CORS_HEADERS'] = 'Content-Type'

x = open('./models/modelGender.json', 'r')
genderModel = x.read()
x.close()
genderModel = model_from_json(genderModel)
genderModel.load_weights("./models/modelGender.h5")

x = open('./models/modelEthnicity.json', 'r')
ethnicityModel = x.read()
x.close()
ethnicityModel = model_from_json(ethnicityModel)
ethnicityModel.load_weights("./models/modelEthnicity.h5")

x = open('./models/modelAge.json', 'r')
ageModel = x.read()
x.close()
ageModel = model_from_json(ageModel)
ageModel.load_weights("./models/modelAge.h5")


@app.route('/')
def check():
    return {"result": "working"}


def detectFaceOpenCVHaar(faceCascade, frame, inHeight=300, inWidth=0):
    frameOpenCVHaar = frame.copy()
    frameHeight = frameOpenCVHaar.shape[0]
    frameWidth = frameOpenCVHaar.shape[1]
    if not inWidth:
        inWidth = int((frameWidth / frameHeight) * inHeight)

    scaleHeight = frameHeight / inHeight
    scaleWidth = frameWidth / inWidth

    frameOpenCVHaarSmall = cv2.resize(frameOpenCVHaar, (inWidth, inHeight))
    frameGray = cv2.cvtColor(frameOpenCVHaarSmall, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(frameGray)
    bboxes = []
    for (x, y, w, h) in faces:
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        cvRect = [int(x1 * scaleWidth), int(y1 * scaleHeight),
                  int(x2 * scaleWidth), int(y2 * scaleHeight)]
        bboxes.append(cvRect)
        cv2.rectangle(frameOpenCVHaar,
                      (cvRect[0], cvRect[1]),
                      (cvRect[2], cvRect[3]),
                      (244, 133, 66),
                      2)
    return frameOpenCVHaar, bboxes


@app.route('/findresults', methods=['POST'])
@cross_origin(supports_credentials=True)
def process():
    file = request.get_json()['image']
    decoded_data = base64.b64decode(file.split(',')[1])
    np_data = np.frombuffer(decoded_data, np.uint8)
    img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    h, w, c = img.shape
    dim = h if h < w else w
    image = cv2.resize(img[:dim, :dim], (500, 500))

    faceCascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

    outOpencvHaar, bboxes = detectFaceOpenCVHaar(faceCascade, image)
    _, encoded_img_main = cv2.imencode('.jpg', outOpencvHaar)
    base64_img_main = base64.b64encode(encoded_img_main).decode("utf-8")
    result = {
        "image": base64_img_main,
        "result": []
    }

    for i, face in enumerate(bboxes):
        face = image[face[1]:face[3], face[0]:face[2]]
        temp_x = cv2.resize(face, (48, 48))
        x = np.reshape(cv2.split(temp_x)[0], newshape=(1, 48, 48, 1))
        gender = genderModel.predict(x)
        eth = ethnicityModel.predict(x)

        age = ageModel.predict(np.reshape(cv2.split(np.array([i / 255.0 for i in temp_x]))[0], newshape=(1, 48, 48, 1)))
        _, encoded_img = cv2.imencode('.jpg', face)
        base64_img = base64.b64encode(encoded_img).decode("utf-8")
        result['result'].append(
            {
                'image': base64_img,
                'gender': gender[0][0],
                'ethnicity': np.argmax(eth),
                'age': age[0][0]
            })
    return str({"result": result}).replace('\'', '\"')


if __name__ == '__main__':
    app.run()
