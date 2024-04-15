#!/home/mark/miniconda3/envs/yolox-ti/bin/python3
import threading
import rospy
import onnxruntime
import rospkg
import os
import numpy as np
import cv2
import time

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage

br = CvBridge()

# taken from P matrix published by /zed2i/zed_node/depth/camera_info
fx = 487.29986572265625
fy = 487.29986572265625
cx = 325.617431640625
cy = 189.09378051757812


def preprocess_rgb(img, input_size, swap=(2, 0, 1)):
    assert len(img.shape) == 3

    # initialize padded image with "neutral" color 114
    padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)

    # place the resized image in top left corner of padded_img
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    # change from (height, width, channels) to (channels, height, width)
    padded_img = padded_img.transpose(swap)

    # ensure mem is contig. for performance reasons
    return np.ascontiguousarray(padded_img, dtype=np.float32), r


class CameraSubscriber:
    def __init__(self):
        package_path = rospkg.RosPack().get_path("treedet_ros")
        model_path = os.path.join(package_path, "model.onnx")
        self.session = onnxruntime.InferenceSession(model_path)
        print("Loaded Model")

        self.data_buffer = ([], [])  # RGB, Depth

        # lock prevents simulataneous R/W to the buffer
        self.lock = threading.Lock()

        self.rgb_subscriber = rospy.Subscriber(
            "/zed2i/zed_node/rgb/image_rect_color/compressed",
            CompressedImage,
            self.rgb_callback,
        )

        self.rgb_subscriber = rospy.Subscriber(
            "/zed2i/zed_node/depth/depth_registered",
            Image,
            self.depth_callback,
        )
        # Timer to process messages at a desired frequency (e.g., 1 Hz)
        self.timer = rospy.Timer(rospy.Duration(1 / 1), self.timer_callback)

    def rgb_callback(self, comp_image: CompressedImage):
        with self.lock:
            self.data_buffer[0].append(comp_image)

    def depth_callback(self, img: Image):
        with self.lock:
            self.data_buffer[1].append(img)

    def timer_callback(self, event):
        with self.lock:
            if self.data_buffer[0] and self.data_buffer[1]:
                # Process the last message received
                rgb = self.data_buffer[0][-1]
                depth = self.data_buffer[1][-1]

                self.process_imgs(rgb, depth)
                self.data_buffer = ([], [])

    def get_detections(self, raw_dets: list, rescale_ratio: float):
        # filter uncertain bad detections
        raw_dets = raw_dets[raw_dets[:, 4] >= 0.95]

        # rescale bbox and kpts w.r.t original image
        raw_dets[:, :4] /= rescale_ratio
        raw_dets[:, 6::3] /= rescale_ratio
        raw_dets[:, 7::3] /= rescale_ratio

        # kpts in format "kpC", "kpL", "kpL", "ax1", "ax2" and each kpt gets (x, y, conf)
        return raw_dets[:, :4], raw_dets[:, 4], raw_dets[:, 6:]

    def process_imgs(self, rgb_msg: CompressedImage, depth_msg: Image):
        rgb_img = br.compressed_imgmsg_to_cv2(rgb_msg)
        depth_img = br.imgmsg_to_cv2(depth_msg)

        height, width = depth_img.shape

        rgb_img = rgb_img[:, :, :3]  # cut out the alpha channel (bgra8 -> bgr8)

        assert (
            rgb_img.shape[0] == depth_img.shape[0]
            and rgb_img.shape[1] == depth_img.shape[1]
        ), f"Invalid image shape rgb: {rgb_img.shape} d: {depth_img.shape}"

        start_time = time.perf_counter()
        img, ratio = preprocess_rgb(rgb_img, (384, 672))

        print(f"preproc:\t{round((time.perf_counter() - start_time) * 1000, 3)} ms")

        # pass image through model
        ort_inputs = {self.session.get_inputs()[0].name: img[None, :, :, :]}
        output = self.session.run(None, ort_inputs)

        print(
            f"preproc + inf:\t{round((time.perf_counter() - start_time) * 1000, 3)} ms"
        )

        bboxes, conf, kpts = self.get_detections(output[0], ratio)
        # print(kpts)

        kpC = np.round(kpts[:, 0:2])

        # discard all coords with invalid indices
        kpC = kpC[(kpC[:, 0] >= 0) & (kpC[:, 1] >= 0)]
        kpC = kpC[(kpC[:, 0] < width) & (kpC[:, 1] < height)]

        # fetch depth data with kpts
        Z = depth_img[kpC[:, 1].astype(int), kpC[:, 0].astype(int)]

        print(Z.transpose())

        # Z =
        # assert False
        # https://docs.opencv.org/4.5.5/pinhole_camera_model.png
        # https://support.stereolabs.com/hc/en-us/articles/4554115218711-How-can-I-convert-3D-world-coordinates-to-2D-image-coordinates-and-viceversa


if __name__ == "__main__":

    rospy.init_node("treedet_inference", anonymous=True)
    rcs = CameraSubscriber()
    rospy.spin()
