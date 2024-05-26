import rospy
import pandas as pd

from harveri_msgs.msg import HarveriDetectedTrees


detected = pd.DataFrame()


def detected_trees_callback(dets):
    global detected

    for tree in dets.trees:
        data = {
            # "id": tree.id,
            "pos_x": tree.x,
            "pos_y": tree.y,
            "pos_z": tree.z,
            "dim_x": tree.dim_x,
            "dim_y": tree.dim_y,
            "dim_z": tree.dim_z,
        }

        # update row if already exists, else create
        if tree.id in detected.index:
            detected.loc[tree.id] = data
        else:
            print(f"added tree with ID {tree.id}")
            data["id"] = tree.id
            data["frame"] = "map"
            new_row = pd.DataFrame([data]).set_index("id")
            detected = pd.concat([detected, new_row])
    return


def shutdown_hook():
    global detected
    print("\nSaving detected trees")
    detected.to_csv("tree_detections.csv")
    print("Saving complete.")


if __name__ == "__main__":
    rospy.init_node("treedet_detection_subscriber")

    rospy.Subscriber(
        "/treedet/detected_trees", HarveriDetectedTrees, detected_trees_callback
    )
    rospy.on_shutdown(shutdown_hook)
    rospy.spin()
