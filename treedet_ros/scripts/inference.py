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
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2

# process incoming images at given frequency:
RATE_LIMIT = 10.0


br = CvBridge()

# taken from P matrix published by /zed2i/zed_node/depth/camera_info
fx = 487.29986572265625
fy = 487.29986572265625
cx = 325.617431640625
cy = 189.09378051757812

point_pub = rospy.Publisher("/tree_det/felling_cut", PointCloud2, queue_size=10)


def preprocess_rgb(img: np.ndarray, input_size: tuple, swap=(2, 0, 1)):
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


def get_detections(raw_dets: list, rescale_ratio: float):
    """
    Get filtered detections in scaling of original rgb and depth image
    """
    # filter uncertain bad detections
    raw_dets = raw_dets[raw_dets[:, 4] >= 0.95]

    # rescale bbox and kpts w.r.t original image
    raw_dets[:, :4] /= rescale_ratio
    raw_dets[:, 6::3] /= rescale_ratio
    raw_dets[:, 7::3] /= rescale_ratio

    # kpts in format "kpC", "kpL", "kpL", "ax1", "ax2" and each kpt gets (x, y, conf)
    return raw_dets[:, :4], raw_dets[:, 4], raw_dets[:, 6:]


def uv2xy(uv: np.ndarray, Z: np.ndarray):
    """Project
    https://support.stereolabs.com/hc/en-us/articles/4554115218711-How-can-I-convert-3D-world-coordinates-to-2D-image-coordinates-and-viceversa
    """
    assert uv.shape[0] == Z.shape[0]

    X = ((uv[:, 0] - cx) * Z) / (fx)
    Y = ((uv[:, 1] - cy) * Z) / (fy)
    return X, Y


def uv2xyz(uv: np.ndarray, depth_img: np.ndarray, height: int, width: int):
    """Convert from pixel coordinates to 3D point with depth image information

    Resulting cartesian coordinates are consistent with the stereolabs frame of ref:
    - https://docs.opencv.org/4.5.5/pinhole_camera_model.png
    - https://www.stereolabs.com/docs/positional-tracking/coordinate-frames#selecting-a-coordinate-system
    """
    positive = (uv[:, 0] >= 0) & (uv[:, 1] >= 0)
    in_bounds = (uv[:, 0] < width) & (uv[:, 1] < height)

    # discard all coords with invalid indices
    uv = uv[positive & in_bounds]

    # fetch depth data in meters
    Z = depth_img[
        uv[:, 1].astype(int),
        uv[:, 0].astype(int),
    ]
    valid_depth = ~(np.isnan(Z) | np.isinf(Z))
    Z = Z[valid_depth]

    X, Y = uv2xy(uv[valid_depth], Z)

    # , positive & in_bounds & valid_z
    return np.column_stack((X, Y, Z)), positive & in_bounds, valid_depth


def uv2roll(uv1: np.ndarray, uv2: np.ndarray) -> np.ndarray:
    du = uv1[:, 0] - uv2[:, 0]
    dv = uv1[:, 1] - uv2[:, 1]

    print(np.arctan2(dv, du) + np.pi / 2)


def pc2_msg(XYZ: np.ndarray) -> PointCloud2:
    header = rospy.Header(
        frame_id="zed2i_left_camera_optical_frame", stamp=rospy.Time.now()
    )
    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
    ]
    points = [tuple(point) for point in XYZ]

    return point_cloud2.create_cloud(header, fields, points)


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
        self.timer = rospy.Timer(rospy.Duration(1 / RATE_LIMIT), self.timer_callback)

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
                rgb_msg: CompressedImage = self.data_buffer[0][-1]
                depth_msg: Image = self.data_buffer[1][-1]

                rgb_img: np.ndarray = br.compressed_imgmsg_to_cv2(rgb_msg)
                depth_img: np.ndarray = br.imgmsg_to_cv2(depth_msg)

                rgb_img = rgb_img[:, :, :3]  # cut out the alpha channel (bgra8 -> bgr8)

                self.process_imgs(rgb_img, depth_img)
                self.data_buffer = ([], [])

    def process_imgs(self, rgb_img: np.ndarray, depth_img: np.ndarray):
        assert (
            rgb_img.shape[0] == depth_img.shape[0]
            and rgb_img.shape[1] == depth_img.shape[1]
        ), f"Invalid image shape rgb: {rgb_img.shape} d: {depth_img.shape}"

        height, width = depth_img.shape

        start_time = time.perf_counter()
        img, ratio = preprocess_rgb(rgb_img, (384, 672))

        print(f"preproc:\t{round((time.perf_counter() - start_time) * 1000, 3)} ms")

        # pass image through model
        ort_inputs = {self.session.get_inputs()[0].name: img[None, :, :, :]}
        output = self.session.run(None, ort_inputs)

        print(
            f"preproc + inf:\t{round((time.perf_counter() - start_time) * 1000, 3)} ms"
        )

        bboxes, conf, kpts = get_detections(output[0], ratio)

        cut_uv = np.round(kpts[:, 0:2])

        # convert felling cut pixel coords to 3D
        cut_XYZ, uv_mask, depth_mask = uv2xyz(cut_uv, depth_img, height, width)

        # # only calculate angle and diameter for kpts that were detected in depth camera
        l_uv = np.round(kpts[:, 3:5])[uv_mask][depth_mask]
        r_uv = np.round(kpts[:, 6:8])[uv_mask][depth_mask]

        # # Reuse the z from the felling cut in trunk center because sometimes depth measurement is more reliable.
        # # Sometimes treedet misses the trunk and then the Z value is in the background, so possibly np.inf or np.nan.
        # # That makes it impossible to use uv2xyz. Instead we project uv onto Z=Z_c plane of the felling cut.
        z_of_cut = cut_XYZ[:, 2]

        assert l_uv.shape == r_uv.shape and r_uv.shape[0] == z_of_cut.shape[0]

        Xl, Yl = uv2xy(l_uv, z_of_cut)
        Xr, Yr = uv2xy(r_uv, z_of_cut)

        diam = np.sqrt((Xr - Xl) ** 2 + (Yr - Yl) ** 2)
        print(diam)

        # ax1_uv = np.round(kpts[:, 9:11])
        # roll = uv2roll(ax1_uv[mask], cut_uv[mask])

        point_pub.publish(pc2_msg(cut_XYZ))


if __name__ == "__main__":
    rospy.init_node("treedet_inference", anonymous=True)
    rcs = CameraSubscriber()
    rospy.spin()
