from segmentations import test as seg
from traffic_object import detect as obj
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from flask import Flask, Response

# Use a service account.
cred = credentials.Certificate('khkys22-firebase-adminsdk-gxlc0-aea51764b0.json')

firebase_app = firebase_admin.initialize_app(cred)

db = firestore.client()

flask_app = Flask(__name__)


def semantic_segmentation():
    enet_opt = seg.Seg(weights='segmentations/models/ckpt-enet-1.pth', num_classes=12, resize_height=512,
                       resize_width=512, device='cpu')

    enet_opt.detect('sample/test.mp4')
    # enet_opt.detect_img('sample/test.png')


cam_urls = ['https://www.youtube.com/watch?v=fiWopDJ3rCs', 'https://www.youtube.com/watch?v=3zH2GwsexiE',
            'https://www.youtube.com/watch?v=FxGndPkyAaQ']

yolo_opt = []

for url in cam_urls:
    yolo_opt.append(obj.YoloOption(weights='traffic_object/best100.pt', conf_thres=0.1, nosave=True,
                                   view_img=False, return_result=True, webcam=True,
                                   source=url,
                                   traced_model='traffic_object/traced_model.pt', update_cloud=True))

@flask_app.route('/cam/<id>')
def video_feed(id):
    # print(yolo_opt[int(id)].source)
    return Response(obj.detect(yolo_opt[int(id)], db),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# flask_app.run(host='127.0.0.1', debug=True)

# import pafy
#
# url = "https://www.youtube.com/watch?v=fiWopDJ3rCs"
# video = pafy.new(url)
# best = video.getbest(preftype="mp4")
#
# cap = cv.VideoCapture(best.url)
# while True:
#     grabbed, frame = cap.read()
#     cv.imshow("Video", frame)
#     w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
#     h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv.CAP_PROP_FPS) % 100
#     print(f' success ({w}x{h} at {fps:.2f} FPS).')
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break
