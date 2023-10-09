import numpy as np
import torch
import open3d as o3d
import frustum_pointnets

# Load the Frustum PointNets model
model = frustum_pointnets.FrustumPointNets()
model.load_state_dict(torch.load('frustum_pointnets.pt'))

# Load the LiDAR point cloud data
lidar_points = np.load('lidar.npy')

# Extract frustums from the LiDAR point cloud data
frustums = frustum_pointnets.extract_frustums(lidar_points)

# Predict the detected objects for each frustum
detections = model(frustums)

# Visualize the detected objects
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add the LiDAR point cloud to the visualizer
vis.add_geometry(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(lidar_points)))

# Add the detected objects to the visualizer
for detection in detections:
    bbox = detection['bbox']
    confidence = detection['confidence']

    # Create a 3D bounding box from the 2D bounding box
    bbox_3d = frustum_pointnets.project_bbox_to_3d(bbox)

    # Add the 3D bounding box to the visualizer
    vis.add_geometry(o3d.geometry.AxisAlignedBoundingBox(bbox_3d, o3d.Vector3d(0, 1, 0)))

# Render the visualizer
vis.update_renderer()
vis.run()

# Save the detection results to a file
np.save('detections.npy', detections)
