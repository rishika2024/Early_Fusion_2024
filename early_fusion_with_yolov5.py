#!/usr/bin/env python3
import rospy
import torch
import yaml
import cv2
import numpy as np
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2

# Global variables
bridge = CvBridge()
camera_image = None
lidar_points = None

# Load calibration file
def load_calibration(file_path):
    with open(file_path, "r") as f:
        calibration = yaml.safe_load(f)
    P2 = np.array(calibration["P2"])
    Tr_velo_to_cam = np.array(calibration["Tr_velo_to_cam"])
    R0_rect = np.array(calibration["R0_rect"])
    return P2, Tr_velo_to_cam, R0_rect

# Callback for Camera Image
def camera_callback(data):
    global camera_image
    camera_image = bridge.imgmsg_to_cv2(data, "bgr8")

# Callback for LiDAR PointCloud
def lidar_callback(data):
    global lidar_points
    lidar_points = np.array(list(pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z"))))

# Project LiDAR points into the camera frame
def project_lidar_to_image(lidar_points, P2, Tr_velo_to_cam, R0_rect):
    lidar_points_hom = np.hstack((lidar_points, np.ones((lidar_points.shape[0], 1))))  # Homogeneous coords
    camera_points = np.dot(R0_rect, np.dot(Tr_velo_to_cam, lidar_points_hom.T)).T
    valid_indices = camera_points[:, 2] > 0
    camera_points = camera_points[valid_indices]
    camera_points_hom = np.hstack((camera_points, np.ones((camera_points.shape[0], 1))))
    image_points_hom = np.dot(P2, camera_points_hom.T).T
    image_points_hom[:, 0] /= image_points_hom[:, 2]
    image_points_hom[:, 1] /= image_points_hom[:, 2]
    image_points = image_points_hom[:, :2]
    return image_points, camera_points

# Overlay LiDAR points on the camera image
def overlay_lidar_on_image(image, image_points, lidar_points):
    for i in range(image_points.shape[0]):
        x, y = int(image_points[i, 0]), int(image_points[i, 1])
        depth = lidar_points[i, 2]
        color = (0, int(255 * (1 - depth / 20.0)), int(255 * (depth / 20.0)))  # Depth-based coloring
        cv2.circle(image, (x, y), 3, color, -1)
    return image

# Use YOLOv5 to detect objects
def detect_objects_yolo(image, yolo_model):
    results = yolo_model(image)  # Perform detection
    detected_objects = results.pandas().xyxy[0]  # Extract bounding box predictions
    return detected_objects

# Overlay YOLO bounding boxes on the image
def overlay_yolo_boxes(image, detections):
    for _, row in detections.iterrows():
        x1, y1, x2, y2 = map(int, row[['xmin', 'ymin', 'xmax', 'ymax']])
        label = f"{row['name']} {row['confidence']:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def main():
    rospy.init_node("early_fusion", anonymous=True)

    # Topics
    camera_topic = "/front/image_raw"
    lidar_topic = "/mid/points"

    # Calibration file
    calibration_file = "/home/rishika/catkin_ws/src/lidar_camera_calibration/calibration_matrix.yaml"

    # Load calibration parameters
    P2, Tr_velo_to_cam, R0_rect = load_calibration(calibration_file)

    # Load YOLOv5 model
    yolo_model = torch.hub.load(
        'ultralytics/yolov5', 'custom', 
        path='/home/rishika/catkin_ws/src/yolo_training/yolov5/runs/train/exp/weights/best.pt'
    )

    # Subscribers
    rospy.Subscriber(camera_topic, Image, camera_callback)
    rospy.Subscriber(lidar_topic, PointCloud2, lidar_callback)
    
    # Main loop
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        if camera_image is not None and lidar_points is not None:
            # Project LiDAR points to the camera frame
            image_points, valid_lidar_points = project_lidar_to_image(lidar_points, P2, Tr_velo_to_cam, R0_rect)

            # Overlay LiDAR points on the camera image
            fusion_image = overlay_lidar_on_image(camera_image.copy(), image_points, valid_lidar_points)

            # Perform object detection using YOLOv5
            yolo_detections = detect_objects_yolo(camera_image, yolo_model)

            # Overlay YOLO detections on the fused image
            fusion_image = overlay_yolo_boxes(fusion_image, yolo_detections)

            # Display the fusion image
            cv2.imshow("Early Fusion with YOLOv5", fusion_image)

            # Keep the OpenCV window responsive; exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        rate.sleep()

    # Release resources
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
