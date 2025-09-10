/*
 * Pointcloud Clustering Detector Cuda
 * Copyright (c) 2025 Menghao Woods
 *
 * Licensed under the MIT License. See LICENSE file in the project root for details.
 */
#include "point_cloud_visualizer.h"

PointCloudVisualizer::PointCloudVisualizer(const YAML::Node &config)
{
    // Initialize the PCL visualizer with the configuration settings
    camera_distance_ = config["point_cloud_visualizer"]["camera_distance"].as<float>(25.0f);
    angle_ = config["point_cloud_visualizer"]["angle"].as<float>(160.0f);
    refresh_interval_ = config["point_cloud_visualizer"]["refresh_interval"].as<int>(10);

    // Log the initialization parameters
    std::cout << "[PointCloudVisualizer] Initializing with parameters:" << std::endl;
    std::cout << "  - Camera Distance: " << camera_distance_ << std::endl;
    std::cout << "  - Angle: " << angle_ << " degrees" << std::endl;
    std::cout << "  - Refresh Interval: " << refresh_interval_ << " ms" << std::endl;

    startViewerThread();    // launch viewer thread
}

PointCloudVisualizer::~PointCloudVisualizer()
{
    stopViewerThread();
}

void PointCloudVisualizer::startViewerThread()
{
    thread_ = std::thread(&PointCloudVisualizer::runViewerLoop, this);
}

void PointCloudVisualizer::stopViewerThread()
{
    // Gracefully shut down the viewer thread.
    {
        std::lock_guard lk(mtx_);
        shutdown_ = true;
    }
    cv_.notify_one(); // Wake up the thread if it's waiting.
    if (thread_.joinable())
    {
        thread_.join();
    }
}

void PointCloudVisualizer::pushFrame(PointCloud::Ptr cloud,
                                     std::vector<float> plane)
{
    {
        std::lock_guard lk(mtx_);
        latest_data_ = VisualizerData{std::move(cloud), std::move(plane)};
    }
    cv_.notify_one();   // Notify the runViewerLoop that new data is available.
}

void PointCloudVisualizer::pushFrame(PointCloud::Ptr cloud)
{
    // Call the original function with an empty vector for the plane
    pushFrame(std::move(cloud), {});
}

void PointCloudVisualizer::pushClusteredFrame(PointCloudPtr cloud,
                                              std::vector<float> plane,
                                              const std::vector<PointCloudPtr>& clusters)
{
    {
        std::lock_guard lk(mtx_);
        // Create a copy of the clusters to avoid data races
        std::vector<PointCloudPtr> clusters_copy;
        for (const auto& cluster : clusters) {
            clusters_copy.push_back(PointCloudPtr(new PointCloud(*cluster)));
        }
        latest_data_ = VisualizerData{std::move(cloud), std::move(plane), std::move(clusters_copy)};
    }
    cv_.notify_one();
}


void PointCloudVisualizer::runViewerLoop()
{
    initViewer();

    // --- Create all necessary placeholder actors once ---
    PointCloud::Ptr dummy(new PointCloud);

    // Actor for the single, unsplit cloud
    viewer_->addPointCloud<Point>(dummy, "cloud");
    viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 1, 1, "cloud"); // White
    viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");

    // Actor for points above the plane
    viewer_->addPointCloud<Point>(dummy, "above_cloud");
    viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 1, 1, "above_cloud"); // White
    viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "above_cloud");

    // Actor for points below the plane
    viewer_->addPointCloud<Point>(dummy, "below_cloud");
    viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 0, "below_cloud"); // Green
    viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "below_cloud");

    // --- Main Update Loop ---
    while (!viewer_->wasStopped() && !shutdown_)
    {
        VisualizerData msg;
        {
            std::unique_lock lk(mtx_);
            cv_.wait(lk, [&]{ return latest_data_.has_value() || shutdown_; });
            if (shutdown_) break;
            msg = std::move(*latest_data_);
            latest_data_.reset();
        }

        // Always clear old bounding boxes
        viewer_->removeAllShapes();

        // --- NEW: Check if clusters were provided ---
        if (!msg.clusters.empty())
        {
            // Visualize the ground and non-ground points with colors
            auto [above, below] = splitCloudByPlane(msg.cloud, msg.plane_coeffs);
            viewer_->updatePointCloud<Point>(above, "above_cloud");
            viewer_->updatePointCloud<Point>(below, "below_cloud");
            viewer_->updatePointCloud<Point>(dummy, "cloud"); // Hide the single cloud

            // Draw the bounding boxes for the clusters
            showBoundingBoxes(msg.clusters);
        }
        else if (msg.plane_coeffs.size() == 4)
        {
            // --- Logic for visualizing with a ground plane without clusters ---
            auto [above, below] = splitCloudByPlane(msg.cloud, msg.plane_coeffs);
            viewer_->updatePointCloud<Point>(above, "above_cloud");
            viewer_->updatePointCloud<Point>(below, "below_cloud");
            viewer_->updatePointCloud<Point>(dummy, "cloud");

            pcl::ModelCoefficients coeffs;
            coeffs.values = msg.plane_coeffs;
            viewer_->addPlane(coeffs, "ground_plane");
        }
        else
        {
            // --- Logic for visualizing a single point cloud ---
            viewer_->updatePointCloud<Point>(msg.cloud, "cloud");
            viewer_->updatePointCloud<Point>(dummy, "above_cloud");
            viewer_->updatePointCloud<Point>(dummy, "below_cloud");
        }

        viewer_->spinOnce(refresh_interval_);
    }
}

void PointCloudVisualizer::initViewer()
{
    // Set default point cloud rendering properties
    viewer_.reset(new pcl::visualization::PCLVisualizer("Ground Segmentation Visualizer"));
    // viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
    viewer_->addCoordinateSystem(1.0); // Add coordinate system for reference
    viewer_->initCameraParameters(); // Initialize camera parameters for the viewer

    // Set the camera distance from the origin (adjust as necessary)
    float camera_distance = camera_distance_;
    float angle = angle_;

    // Calculate the camera position based on rotation around the y-axis
    float x_position = camera_distance * cos(angle * M_PI / 180.0f);  // x position after rotation
    float z_position = camera_distance * sin(angle * M_PI / 180.0f);  // z position after rotation
    float y_position = 0.0f;  // Camera height (adjust as necessary)

    // Set the camera position and make it look at the origin (0, 0, 0)
    viewer_->setCameraPosition(x_position, y_position, z_position,  // Camera position (x, y, z)
                               0.0, 0.0, 0.0,                   // Look at the origin (0, 0, 0)
                               1.0, 0.0, 0.0);                  // Up direction along the x-axis
}

std::pair<PointCloud::Ptr, PointCloud::Ptr> PointCloudVisualizer::splitCloudByPlane(
    const PointCloud::Ptr& cloud, const std::vector<float>& plane_coeffs, float threshold)
{
    PointCloud::Ptr above(new PointCloud);
    PointCloud::Ptr below(new PointCloud);

    float a = plane_coeffs[0];
    float b = plane_coeffs[1];
    float c = plane_coeffs[2];
    float d = plane_coeffs[3];

    for (const auto& point : cloud->points)
    {
        float distance = a * point.x + b * point.y + c * point.z + d;
        if (distance > threshold)
            above->points.push_back(point);
        else
            below->points.push_back(point);
    }

    above->width = static_cast<uint32_t>(above->points.size());
    below->width = static_cast<uint32_t>(below->points.size());
    above->height = below->height = 1;
    above->is_dense = below->is_dense = true;

    return {above, below};
}

void PointCloudVisualizer::showBoundingBoxes(const std::vector<PointCloudPtr>& clusters)
{
    int i = 0;
    for (const auto& cluster : clusters)
    {
        if (cluster->points.empty()) continue;

        // Get the min and max points to define the bounding box
        Point min_pt, max_pt;
        pcl::getMinMax3D(*cluster, min_pt, max_pt);

        // Define a unique ID for the bounding box
        std::string bbox_id = "bbox_" + std::to_string(i);

        // Add the bounding box to the visualizer
        viewer_->addCube(min_pt.x, max_pt.x, min_pt.y, max_pt.y, min_pt.z, max_pt.z,
                         1.0, 0.0, 0.0, bbox_id); // Red bounding box
        viewer_->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
                                            pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME,
                                            bbox_id);
        viewer_->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3.0, bbox_id);
        i++;
    }
}
