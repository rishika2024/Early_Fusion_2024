#!/usr/bin/env python3
import rospy
import tf
import yaml
import numpy as np

def get_camera_intrinsics():
    """
    Simulated intrinsics for the Pointgrey camera.
    If the intrinsics are available in ROS parameters, use them instead.
    """
    # Replace with actual parameters if known
    camera_intrinsics = {
        "fx": 554.3827,  # Focal length x
        "fy": 554.3827,  # Focal length y
        "cx": 320.5,     # Principal point x
        "cy": 240.5      # Principal point y
    }
    P = np.array([
        [camera_intrinsics["fx"], 0, camera_intrinsics["cx"], 0],
        [0, camera_intrinsics["fy"], camera_intrinsics["cy"], 0],
        [0, 0, 1, 0]
    ])
    return P

def get_lidar_to_camera_transform(lidar_frame, camera_frame):
    """
    Retrieves the LiDAR-to-Camera transformation from the /tf topic.
    """
    listener = tf.TransformListener()
    rospy.loginfo(f"Waiting for transform from {lidar_frame} to {camera_frame}...")
    listener.waitForTransform(camera_frame, lidar_frame, rospy.Time(), rospy.Duration(5.0))

    try:
        (trans, rot) = listener.lookupTransform(camera_frame, lidar_frame, rospy.Time(0))
        T = tf.transformations.translation_matrix(trans)  # Translation matrix
        R = tf.transformations.quaternion_matrix(rot)     # Rotation matrix
        transform_matrix = np.dot(T, R)  # Combine translation and rotation into a 4x4 matrix
        return transform_matrix[:3, :]  # Return 3x4 extrinsic calibration matrix
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
        rospy.logerr(f"Error while fetching transform: {e}")
        return None

def save_calibration_file(output_path, camera_intrinsics, extrinsics):
    """
    Saves the calibration matrix to a YAML file.
    """
    calibration_data = {
        "P2": camera_intrinsics.tolist(),  # Camera intrinsics matrix
        "Tr_velo_to_cam": extrinsics.tolist(),  # LiDAR to Camera extrinsics
        "R0_rect": np.eye(3).tolist()  # Rectification matrix (identity in most cases)
    }
    with open(output_path, "w") as outfile:
        yaml.dump(calibration_data, outfile, default_flow_style=False)
    rospy.loginfo(f"Calibration data saved to {output_path}")

if __name__ == "__main__":
    rospy.init_node("generate_calibration_matrix", anonymous=True)

    # Specify the LiDAR and Camera frames
    lidar_frame = "mid_vlp16_mount"  # Frame of your 3D LiDAR
    camera_frame = "front_camera_optical"  # Optical frame of the camera

    # Get Camera Intrinsics
    camera_intrinsics = get_camera_intrinsics()

    # Get Extrinsics (LiDAR-to-Camera Transformation)
    extrinsics = get_lidar_to_camera_transform(lidar_frame, camera_frame)
    if extrinsics is None:
        rospy.logerr("Failed to get LiDAR-to-Camera transform!")
        exit()

    # Save the calibration data to a YAML file
    output_path = "/tmp/calibration_matrix.yaml"
    save_calibration_file(output_path, camera_intrinsics, extrinsics)
