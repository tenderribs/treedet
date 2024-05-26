import open3d as o3d
import numpy as np


def load_point_cloud(file_path):
    return o3d.io.read_point_cloud(file_path)


def save_point_cloud(point_cloud, file_path):
    o3d.io.write_point_cloud(file_path, point_cloud)


def crop_point_cloud(point_cloud, min_bound, max_bound):
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    cropped_point_cloud = point_cloud.crop(bounding_box)
    return cropped_point_cloud


def main():
    input_pcd_file = "/datasets/maps/straight_line.pcd"
    output_pcd_file = "/datasets/maps/straight_line_cropped.pcd"

    # Define the bounding box (min and max points in x, y, z)
    min_bound = [-150, -138, -40.0]
    max_bound = [-100, -73, 40.0]

    # Load the point cloud
    point_cloud = load_point_cloud(input_pcd_file)

    # Crop the point cloud
    cropped_point_cloud = crop_point_cloud(point_cloud, min_bound, max_bound)

    # Save the cropped point cloud
    save_point_cloud(cropped_point_cloud, output_pcd_file)
    print(f"Cropped point cloud saved to {output_pcd_file}")


if __name__ == "__main__":
    main()
