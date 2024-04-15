#!/usr/bin/env python
import threading
import rospy

from sensor_msgs.msg import CompressedImage

class RateControlledSubscriber:
    def __init__(self):
        self.data_buffer = []

        # lock prevents simulataneous R/W to the buffer
        self.lock = threading.Lock()

        # Set up subscriber
        self.subscriber = rospy.Subscriber("/zed2i/zed_node/rgb/image_rect_color/compressed", CompressedImage, self.callback)

        # Timer to process messages at a desired frequency (e.g., 1 Hz)
        self.timer = rospy.Timer(rospy.Duration(1.0), self.timer_callback)

    def callback(self, data):
        with self.lock:
            self.data_buffer.append(data)

    def timer_callback(self, event):
        with self.lock:
            if self.data_buffer:
                # Process the last message received
                message_to_process = self.data_buffer[-1]
                self.process_data(message_to_process)
                # Clear the buffer
                self.data_buffer = []

    def process_data(self, data):
        rospy.loginfo("Processing data")

if __name__ == '__main__':
    rospy.init_node('rate_controlled_subscriber', anonymous=True)
    rcs = RateControlledSubscriber()
    rospy.spin()