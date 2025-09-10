/*
 * Pointcloud Clustering Detector Cuda
 * Copyright (c) 2025 Menghao Woods
 *
 * Licensed under the MIT License. See LICENSE file in the project root for details.
 */
#include "constant.h"

#include <chrono>
#include <thread>
#include <vector>

#include "point_cloud_loader.h"
#include "point_cloud_visualizer.h"
#include "box_filter.h"
#include "voxel_filter.h"
#include "noise_filter.h"
#include "ground_estimation.h"
#include "object_detector.h" // New include

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <config_path>" << std::endl;
        return 1;
    }
    std::string config_path = argv[1];

    // Load configuration
    YAML::Node config = YAML::LoadFile(config_path);
    bool verbose = config["verbose"].as<bool>(false);
    bool timer = config["timer"].as<bool>(false);
    float frequency = config["frequency"].as<float>(10.0f); // Default frequency is 10 Hz
    bool visualization = config["point_cloud_visualizer"]["enable"].as<bool>(true);
    bool ground_estimation = config["ground_estimation"]["enable"].as<bool>(true);
    bool object_detection = config["object_detector"]["enable"].as<bool>(true); // New config option

    auto target_frame_duration = std::chrono::duration<double>(1.0 / frequency);

    // Logs
    std::cout << "Verbose: " << (verbose ? "true" : "false") << std::endl;
    std::cout << "Timer: " << (timer ? "true" : "false") << std::endl;
    std::cout << "Frequency: " << frequency << " Hz" << std::endl;
    std::cout << "Target frame duration: " << target_frame_duration.count() * 1000 << " ms" << std::endl;
    std::cout << "visualization: " << (visualization ? "true" : "false") << std::endl;
    std::cout << "ground_estimation: " << (ground_estimation ? "true" : "false") << std::endl;
    std::cout << "object_detection: " << (object_detection ? "true" : "false") << std::endl;

    std::cout << "Waiting for 3 seconds before starting..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(3));

    // Initialize components
    PointCloudLoader point_cloud_loader(config);
    PointCloudVisualizer point_cloud_visualizer(config);
    // Filters for ground estimation
    BoxFilter box_filter_ransac(config["box_filter_ransac"]);
    VoxelFilter voxel_filter_ransac(config["voxel_filter_ransac"]);
    NoiseFilter noise_filter(config);
    GroundEstimation ground_estimator(config);
    // Object Detector
    BoxFilter box_filter_detector(config["box_filter_detector"]);
    VoxelFilter voxel_filter_detector(config["voxel_filter_detector"]);
    ObjectDetector object_detector(config); // New component

    PointCloudPtr cloud(new PointCloud());
    std::vector<float> ground_plane;

    while (point_cloud_loader.loadNextPointCloud(cloud))
    {
        // Init
        auto frame_start = std::chrono::steady_clock::now();
        if (timer) std::cout << "------------------- Frame -------------------" << std::endl;

        // Create a copy of the point cloud for visualization
        PointCloudPtr cloud_copy(new pcl::PointCloud<pcl::PointXYZI>());
        PointCloudPtr cloud_detector(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::copyPointCloud(*cloud, *cloud_detector); // Copy data from loaded cloud to the copy
        pcl::copyPointCloud(*cloud, *cloud_copy); // Copy data from loaded cloud to the copy

        // Filters
        auto t0 = std::chrono::steady_clock::now();
        box_filter_ransac.applyFilter(cloud);
        voxel_filter_ransac.applyFilter(cloud);
        noise_filter.applyFilter(cloud);

        box_filter_detector.applyFilter(cloud_detector);
        voxel_filter_detector.applyFilter(cloud_detector);

        // ground estimation:  Ransac, wall filter, moving average, or kalman filter
        PointCloudPtr non_ground_cloud(new PointCloud()); // Declare the point cloud here
        bool ground_estimated = ground_estimator.estimateGround(cloud, ground_plane);
        if (ground_estimated)
        {
            ground_estimator.segmentCloud(cloud_detector, ground_plane, non_ground_cloud);
            if (verbose)
            {
                std::cout << "\t[GroundEstimation] Estimated plane coefficients: "
                          << ground_plane[0] << ", " << ground_plane[1] << ", "
                          << ground_plane[2] << ", " << ground_plane[3] << std::endl;
            }
        }
        else
        {
            pcl::copyPointCloud(*cloud, *non_ground_cloud);
            if (verbose)
            {
                std::cout << "\t[GroundEstimation] Failed to estimate ground plane." << std::endl;
            }
        }

        if (timer)
        {
            auto t1 = std::chrono::steady_clock::now();
            std::cout << "[Timer] Ground Estimation: "
                    << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                    << " ms" << std::endl;
            t0 = t1;
        }

        // Object Detection
        if (object_detection)
        {
            if (verbose) std::cout << "\t[ObjectDetector] Starting CPU clustering..." << std::endl;
            auto t_cpu_start = std::chrono::steady_clock::now();
            std::vector<PointCloudPtr> cpu_clusters = object_detector.detectObjectsCPU(non_ground_cloud, ground_plane);
            auto t_cpu_end = std::chrono::steady_clock::now();
            auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_cpu_end - t_cpu_start).count();
            if (1)
            {
                std::cout << "CPU Clustering: " << cpu_duration << " ms" << std::endl;
            }
            std::cout << "\t[ObjectDetector] Found " << cpu_clusters.size() << " objects (CPU)." << std::endl;


            if (verbose) std::cout << "\t[ObjectDetector] Starting CUDA clustering..." << std::endl;
            auto t_cuda_start = std::chrono::steady_clock::now();
            std::vector<PointCloudPtr> cuda_clusters = object_detector.detectObjectsCUDA(non_ground_cloud);
            auto t_cuda_end = std::chrono::steady_clock::now();
            auto cuda_duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_cuda_end - t_cuda_start).count();
            if (1)
            {
                std::cout << "CUDA Clustering: " << cuda_duration << " ms" << std::endl;
            }
            std::cout << "\t[ObjectDetector] Found " << cuda_clusters.size() << " objects (CUDA)." << std::endl;

            // Note: For visualization, you can choose to visualize either CPU or CUDA results
            if (visualization)
            {
                point_cloud_visualizer.pushClusteredFrame(cloud_copy, ground_plane, cpu_clusters);
            }
        }
        else
        {
             // Visualize the point cloud if visualization is enabled
            if (visualization && cloud_copy->size() > 0)
            {
                // Update the point_cloud_visualizer with the current point cloud and/or ground plane
                if (ground_estimated)
                {
                    point_cloud_visualizer.pushFrame(cloud_copy, ground_plane);
                }
                else
                {
                    point_cloud_visualizer.pushFrame(cloud_copy);
                }
            }
        }

        // --- FREQUENCY CONTROL LOGIC --- //
        auto processing_end = std::chrono::steady_clock::now();
        auto processing_duration = processing_end - frame_start;

        if (processing_duration < target_frame_duration)
        {
            auto wait_duration = target_frame_duration - processing_duration;
            std::this_thread::sleep_for(wait_duration);
        }

        if (timer)
        {
            auto frame_end = std::chrono::steady_clock::now();
            auto frame_duration = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end - frame_start).count();
            std::cout << "[Timer] Total Frame Time: " << frame_duration << " ms" << std::endl;
        }
        if (timer) std::cout << "------------------- End ---------------------" << std::endl;
        std::cout << std::flush;
    }

    return 0;
}
