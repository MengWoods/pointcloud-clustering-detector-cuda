/*
 * Pointcloud Clustering Detector Cuda
 * Copyright (c) 2025 Menghao Woods
 *
 * Licensed under the MIT License. See LICENSE file in the project root for details.
 */
#include "voxel_filter.h"

VoxelFilter::VoxelFilter(const YAML::Node &config)
{
    // Check if a specific filter configuration is provided
    if (config["enable"])
    {
        enabled_ = config["enable"].as<bool>(false);
        voxel_size_ = config["voxel_size"].as<float>(0.1f);
    }
    else
    {
        // Default to a disabled filter if no configuration is found
        enabled_ = false;
        voxel_size_ = 0.1f;
    }

    if (enabled_)
    {
        std::cout << "[VoxelFilter] Loaded parameters:" << std::endl;
        std::cout << "  - Enabled: " << (enabled_ ? "true" : "false") << std::endl;
        std::cout << "  - Voxel Size: " << voxel_size_ << std::endl;
    }
    else
    {
        std::cout << "[VoxelFilter] Voxel filter is disabled." << std::endl;
    }
}


void VoxelFilter::applyFilter(PointCloudPtr &cloud)
{
    if (!enabled_)
    {
        return;
    }

    pcl::VoxelGrid<Point> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(voxel_size_, voxel_size_, voxel_size_);

    PointCloudPtr filtered_cloud(new PointCloud);
    sor.filter(*filtered_cloud);

    cloud.swap(filtered_cloud); // Replace input cloud with filtered version

    if (verbose_)
    {
        std::cout << "\t[VoxelFilter] Filtered cloud size: " << cloud->size() << " points." << std::endl;
    }
}
