from urllib.request import urlopen
import cv2 as cv
import numpy as np

from flask import Flask, render_template, Response

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


def gen():
    # change to your ESP32-CAM ip
    url = "http://192.168.1.13/video"
    CAMERA_BUFFRER_SIZE = 4096
    stream = urlopen(url)
    bts = b''
    i = 0
    frame_num = 0
    while True:
        try:
            bts += stream.read(CAMERA_BUFFRER_SIZE)
            jpghead = bts.find(b'\xff\xd8')
            jpgend = bts.find(b'\xff\xd9')
            if jpghead > -1 and jpgend > -1:
                frame_num += 1

                jpg = bts[jpghead:jpgend + 2]
                bts = bts[jpgend + 2:]
                img = cv.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv.IMREAD_UNCHANGED)
                # img=cv.flip(img,0)
                # h,w=img.shape[:2]

                img = cv.resize(img, (640, 480))
                # cv.imshow("a", img)

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n\r\n')

        except Exception as e:
            print("Error:" + str(e))
            bts = b''
            stream = urlopen(url)
            continue

        k = cv.waitKey(1)
        if k & 0xFF == ord('a'):
            cv.imwrite(str(i) + ".jpg", img)
            i = i + 1
        if k & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break


@app.route('/video')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)
