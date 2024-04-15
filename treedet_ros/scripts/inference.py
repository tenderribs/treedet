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
from sensor_msgs.msg import CompressedImage

br = CvBridge()


def preprocess(img, input_size, swap=(2, 0, 1)):
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


class RateControlledSubscriber:
    def __init__(self):
        self.data_buffer = []

        # lock prevents simulataneous R/W to the buffer
        self.lock = threading.Lock()

        self.subscriber = rospy.Subscriber(
            "/zed2i/zed_node/rgb/image_rect_color/compressed",
            CompressedImage,
            self.callback,
        )

        # Timer to process messages at a desired frequency (e.g., 1 Hz)
        self.timer = rospy.Timer(rospy.Duration(1 / 10), self.timer_callback)

        package_path = rospkg.RosPack().get_path("treedet_ros")
        model_path = os.path.join(package_path, "model.onnx")
        self.session = onnxruntime.InferenceSession(model_path)
        print("Loaded Model")

    def callback(self, data):
        with self.lock:
            self.data_buffer.append(data)

    def timer_callback(self, event):
        with self.lock:
            if self.data_buffer:
                # Process the last message received
                message_to_process = self.data_buffer[-1]
                self.process_img(message_to_process)
                self.data_buffer = []

    def process_img(self, data: CompressedImage):
        origin_img = br.compressed_imgmsg_to_cv2(data)
        origin_img = origin_img[:, :, :3]  # cut out the alpha channel (bgra8 -> bgr8)

        start_time = time.perf_counter()
        img, ratio = preprocess(origin_img, (384, 672))

        print(f"preproc: {round((time.perf_counter() - start_time) * 1000, 3)} ms")

        # pass image through model
        ort_inputs = {self.session.get_inputs()[0].name: img[None, :, :, :]}
        output = self.session.run(None, ort_inputs)

        print(f"preproc + inf: {round((time.perf_counter() - start_time) * 1000, 3)} ms")

        dets = output[0]

        # rescale bbox and kpts
        dets[:, :4] /= ratio
        dets[:, 6::3] /= ratio
        dets[:, 7::3] /= ratio

        if dets is not None:
            for det in dets:
                # plot the bounding box
                p1 = (int(det[0]), int(det[1]))
                p2 = (int(det[2]), int(det[3]))
                cv2.rectangle(img, p1, p2, (255, 251, 43), 1)

                # plot the x and y keypoints with sufficient confidence score
                for x, y, conf, label in zip(
                    det[6::3],
                    det[7::3],
                    det[8::3],
                    ["kpC", "kpL", "kpL", "ax1", "ax2"],
                ):
                    cv2.circle(
                        origin_img,
                        (int(x), int(y)),
                        radius=2,
                        color=(52, 64, 235),
                        thickness=-1,
                    )
                    cv2.putText(
                        origin_img,
                        label,
                        (int(x), int(y) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.1,
                        (255, 255, 255),
                        1,
                    )
        cv2.imshow("orig", origin_img)
        cv2.waitKey(2)


if __name__ == "__main__":

    rospy.init_node("treedet_inference", anonymous=True)
    rcs = RateControlledSubscriber()
    rospy.spin()
