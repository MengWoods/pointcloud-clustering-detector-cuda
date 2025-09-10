/*
 * Pointcloud Clustering Detector Cuda
 * Copyright (c) 2025 Menghao Woods
 *
 * Licensed under the MIT License. See LICENSE file in the project root for details.
 */
#pragma once
#include "constant.h"

#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/common/common.h>
#include <vector>
#include <memory>

class ObjectDetector
{
public:
    /**
     * @brief Constructs an ObjectDetector.
     * @param config The YAML configuration node for the detector.
     */
    ObjectDetector(const YAML::Node& config);

    /**
     * @brief Performs CPU-based Euclidean clustering on the non-ground point cloud.
     * @param non_ground_cloud The input point cloud containing non-ground points.
     * @return A vector of point clouds, where each point cloud represents a detected object.
     */
    std::vector<PointCloudPtr> detectObjectsCPU(const PointCloudPtr& non_ground_cloud, const std::vector<float>& ground_plane) const;

    /**
     * @brief Performs CUDA-based clustering on the non-ground point cloud.
     * @param non_ground_cloud The input point cloud containing non-ground points.
     * @return A vector of point clouds, where each point cloud represents a detected object.
     * @note This is a placeholder for a CUDA implementation. A full implementation
     * would require a custom CUDA kernel or a specialized library.
     */
    std::vector<PointCloudPtr> detectObjectsCUDA(const PointCloudPtr& non_ground_cloud) const;

private:
    /**
     * @brief Filters clusters based on bounding box dimensions.
     * @param clusters The input vector of clusters.
     * @return A new vector containing only the filtered clusters.
     */
    std::vector<PointCloudPtr> filterBoundingBoxes(const std::vector<PointCloudPtr>& clusters) const;

        /**
     * @brief Filters clusters that are "flying" or not on the ground.
     * @param clusters A vector of point clouds (clusters).
     * @param ground_plane The estimated ground plane coefficients (A, B, C, D).
     * @return A new vector of point clouds that pass the filter.
     */
    std::vector<PointCloudPtr> filterNonGroundObjects(const std::vector<PointCloudPtr>& clusters, const std::vector<float>& ground_plane) const;


    struct BoundingBoxLimits
    {
        float min_x, max_x;
        float min_y, max_y;
        float min_z, max_z;
    };

    YAML::Node config_;
    float cluster_tolerance_;
    int min_cluster_size_;
    int max_cluster_size_;
};
