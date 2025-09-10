/*
 * Pointcloud Clustering Detector Cuda
 * Copyright (c) 2025 Menghao Woods
 *
 * Licensed under the MIT License. See LICENSE file in the project root for details.
 */
#include "object_detector.h"

ObjectDetector::ObjectDetector(const YAML::Node& config) : config_(config)
{
    // Load clustering parameters from the configuration file
    cluster_tolerance_ = config_["object_detector"]["cluster_tolerance"].as<float>(0.5f);
    min_cluster_size_ = config_["object_detector"]["min_cluster_size"].as<int>(20);
    max_cluster_size_ = config_["object_detector"]["max_cluster_size"].as<int>(25000);
}

std::vector<PointCloudPtr> ObjectDetector::detectObjectsCPU(const PointCloudPtr& non_ground_cloud, const std::vector<float>& ground_plane) const
{
    std::vector<PointCloudPtr> clusters;
    if (non_ground_cloud->empty())
    {
        return clusters;
    }

    // Create a k-d tree representation for the search method
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
    tree->setInputCloud(non_ground_cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
    ec.setClusterTolerance(cluster_tolerance_);
    ec.setMinClusterSize(min_cluster_size_);
    ec.setMaxClusterSize(max_cluster_size_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(non_ground_cloud);
    ec.extract(cluster_indices);

    // Create a new point cloud for each detected cluster
    for (const auto& indices : cluster_indices)
    {
        PointCloudPtr cluster(new PointCloud);
        for (const auto& index : indices.indices)
        {
            cluster->points.push_back(non_ground_cloud->points[index]);
        }
        cluster->width = cluster->points.size();
        cluster->height = 1;
        cluster->is_dense = true;
        clusters.push_back(cluster);
    }

    // First, filter by bounding box size
    auto filtered_by_bbox = filterBoundingBoxes(clusters);

    // Second, filter by ground distance
    return filterNonGroundObjects(filtered_by_bbox, ground_plane);
}

std::vector<PointCloudPtr> ObjectDetector::filterBoundingBoxes(const std::vector<PointCloudPtr>& clusters) const
{
    // Hardcoded bounding box limits for common road users.
    // Dimensions are in meters.
    const std::vector<BoundingBoxLimits> limits = {
        // Bicycle/Motorcycle
        {0.5f, 2.5f, 0.5f, 1.5f, 1.0f, 2.0f},
        // Car
        {2.5f, 5.5f, 1.5f, 2.0f, 1.2f, 1.8f},
        // Van/SUV
        {4.5f, 6.5f, 1.8f, 2.2f, 1.7f, 2.5f},
        // Truck/Bus
        {6.0f, 15.0f, 2.0f, 3.0f, 2.5f, 4.5f}
    };

    std::vector<PointCloudPtr> filtered_clusters;
    pcl::PointXYZI min_pt, max_pt;

    for (const auto& cluster : clusters)
    {
        if (cluster->points.empty())
        {
            continue;
        }

        pcl::getMinMax3D(*cluster, min_pt, max_pt);
        float dx = max_pt.x - min_pt.x;
        float dy = max_pt.y - min_pt.y;
        float dz = max_pt.z - min_pt.z;

        bool is_valid = false;
        for (const auto& limit : limits)
        {
            if (dx >= limit.min_x && dx <= limit.max_x &&
                dy >= limit.min_y && dy <= limit.max_y &&
                dz >= limit.min_z && dz <= limit.max_z)
            {
                is_valid = true;
                break;
            }
        }
        if (is_valid)
        {
            filtered_clusters.push_back(cluster);
        }
    }
    return filtered_clusters;
}

std::vector<PointCloudPtr> ObjectDetector::filterNonGroundObjects(const std::vector<PointCloudPtr>& clusters, const std::vector<float>& ground_plane) const
{
    std::vector<PointCloudPtr> filtered_clusters;
    pcl::PointXYZI min_pt, max_pt;
    const float ground_threshold = 0.5f; // Threshold in meters to be considered "on the ground"

    // If no ground plane is provided, return all clusters.
    if (ground_plane.size() != 4)
    {
        return clusters;
    }

    // Extract plane coefficients
    float a = ground_plane[0];
    float b = ground_plane[1];
    float c = ground_plane[2];
    float d = ground_plane[3];

    for (const auto& cluster : clusters)
    {
        if (cluster->points.empty())
        {
            continue;
        }

        // Find the lowest point in the cluster (min_z)
        pcl::getMinMax3D(*cluster, min_pt, max_pt);
        float cluster_min_z = min_pt.z;

        // Calculate the corresponding ground z-coordinate for the cluster's horizontal center
        // This avoids issues with slanted ground planes
        float cluster_center_x = (min_pt.x + max_pt.x) / 2.0f;
        float cluster_center_y = (min_pt.y + max_pt.y) / 2.0f;

        // Use the plane equation to find the z-value on the plane at the cluster's center
        // ax + by + cz + d = 0  =>  cz = -ax - by - d  => z = (-ax - by - d) / c
        // Handle the case where c is near zero (vertical plane)
        if (std::abs(c) < 1e-6)
        {
            continue; // Cannot determine height relative to a vertical plane
        }
        float ground_z = (-a * cluster_center_x - b * cluster_center_y - d) / c;

        // Check if the cluster's lowest point is near the ground plane
        if (std::abs(cluster_min_z - ground_z) < ground_threshold)
        {
            filtered_clusters.push_back(cluster);
        }
    }
    return filtered_clusters;
}
