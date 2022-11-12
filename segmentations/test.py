from segmentations.utils import *
from segmentations.models.ENet import ENet
import os
import matplotlib.pyplot as plt


class Seg:

    def __init__(self, weights='segmentations/models/ckpt-enet-1.pth', num_classes=12, resize_height=512,
                 resize_width=512, device='cpu', out_path='sample/decoded_segmap.jpg'):
        self.weights = weights
        self.num_classes = num_classes
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.device = device
        self.out_path = out_path
        # Check if the pretrained model is available
        if not self.weights.endswith('.pth'):
            raise RuntimeError('Unknown file passed. Must end with .pth')
        checkpoint = torch.load(self.weights, map_location=self.device)
        # Assuming the dataset is camvid
        self.enet = ENet(self.num_classes)
        self.enet.load_state_dict(checkpoint['state_dict'])

    def detect(self, path):
        vid = cv2.VideoCapture(path)

        while True:

            # Capture the video frame
            # by frame
            ret, frame = vid.read()

            if not ret:
                continue

            # Display the resulting frame

            # if self.image_path is None or not os.path.exists(self.image_path):
            #     raise RuntimeError('An image file path must be passed')

            # tmg_ = plt.imread(self.image_path)
            tmg_ = frame
            # tmg_ = cv2.resize(tmg_, (self.resize_height, self.resize_width), cv2.INTER_NEAREST)
            tmg = torch.tensor(tmg_).unsqueeze(0).float()
            tmg = tmg.transpose(2, 3).transpose(1, 2).to(self.device)

            self.enet.to(self.device)
            with torch.no_grad():
                out1 = self.enet(tmg.float()).squeeze(0)

            # smg_ = Image.open('/content/training/semantic/' + fname)
            # smg_ = cv2.resize(np.array(smg_), (512, 512), cv2.INTER_NEAREST)

            b_ = out1.data.max(0)[1].cpu().numpy()

            decoded_segmap = decode_segmap(b_)

            cv2.imshow('frame', decoded_segmap)

            # images = {
            #     0: ['Input Image', tmg_],
            #     1: ['Predicted Segmentation', b_],
            #     2: ['Decoded Segmentation', decoded_segmap]
            # }
            #
            # show_images(images)

            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def detect_img(self, path):
        checkpoint = torch.load(self.weights, map_location=self.device)

        # Assuming the dataset is camvid
        enet = ENet(self.num_classes)
        enet.load_state_dict(checkpoint['state_dict'])

        tmg_ = plt.imread(path)
        tmg_ = cv2.resize(tmg_, (self.resize_height, self.resize_width), cv2.INTER_NEAREST)
        tmg = torch.tensor(tmg_).unsqueeze(0).float()
        tmg = tmg.transpose(2, 3).transpose(1, 2).to(self.device)

        enet.to(self.device)
        with torch.no_grad():
            out1 = enet(tmg.float()).squeeze(0)

        # smg_ = Image.open('/content/training/semantic/' + fname)
        # smg_ = cv2.resize(np.array(smg_), (512, 512), cv2.INTER_NEAREST)

        b_ = out1.data.max(0)[1].cpu().numpy()

        decoded_segmap = decode_segmap(b_)

        images = {
            0: ['Input Image', tmg_],
            1: ['Predicted Segmentation', b_],
            2: ['Decoded Segmentation', decoded_segmap]
        }

        # show_images(images)

        cv2.imwrite(self.out_path, decoded_segmap)
