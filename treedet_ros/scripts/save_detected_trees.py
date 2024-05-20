import rospy
import pandas as pd

from harveri_msgs.msg import HarveriDetectedTrees, HarveriDetectedTree


def detected_trees_callback(trees: HarveriDetectedTrees):

    return


if __name__ == "__main__":
    rospy.init_node("treedet_detection_subscriber")

    targets = pd.read_csv("trees_found.csv")
    targets = targets[targets["selected"] == True]
    targets["found"] = False
    print(targets)

    rospy.Subscriber(
        "/treedet/detected_trees", HarveriDetectedTrees, detected_trees_callback
    )

    rospy.spin()
