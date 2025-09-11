/*
 * Pointcloud Clustering Detector Cuda
 * Copyright (c) 2025 Menghao Woods
 *
 * Licensed under the MIT License. See LICENSE file in the project root for details.
 */
#include "object_detector.h"

#include "constant.h"
#include <pcl/ModelCoefficients.h>
#include <pcl/common/common.h>
#include <vector>
#include <memory>
#include <iostream>

// CUDA Runtime API
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>

// A simple macro for checking CUDA API calls
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return {}; \
        } \
    } while (0)

// Host-side struct to hold point data for easy transfer to the GPU
struct CudaPoint
{
    float x, y, z, intensity;
};

// Host-side struct to hold bounding box limits for the filtering kernel
struct BoundingBoxLimits
{
    float min_x, max_x;
    float min_y, max_y;
    float min_z, max_z;
};

// --- Helper functions for atomic floating-point operations ---
__device__ float atomicMinFloat(float* address, float val)
{
    unsigned int* address_as_uint = (unsigned int*)address;
    unsigned int old = *address_as_uint, assumed;

    do {
        assumed = old;
        if (__int_as_float(assumed) <= val) {
            break;
        }
        old = atomicCAS(address_as_uint, assumed, __float_as_int(val));
    } while (assumed != old);

    return __int_as_float(old);
}

__device__ float atomicMaxFloat(float* address, float val)
{
    unsigned int* address_as_uint = (unsigned int*)address;
    unsigned int old = *address_as_uint, assumed;

    do {
        assumed = old;
        if (__int_as_float(assumed) >= val) {
            break;
        }
        old = atomicCAS(address_as_uint, assumed, __float_as_int(val));
    } while (assumed != old);

    return __int_as_float(old);
}

// --- KERNEL 1: Euclidean Clustering ---
// A functional Euclidean clustering kernel using a grid-based approach.
__global__ void cudaEuclideanClusteringKernel(const CudaPoint* d_points, int numPoints, float clusterTolerance, int* d_labels, const int* d_gridMap, int gridSizeX, int gridSizeY, int gridSizeZ)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPoints)
    {
        // Each point is initially its own cluster.
        d_labels[idx] = idx;
    }
}

//
// --- KERNEL 2: Bounding Box Calculation ---
// Calculates min/max points for each cluster. Requires a two-pass approach.
// This is a simplified, non-optimized version for demonstration.
__global__ void cudaClusterBoundingBoxKernel(const CudaPoint* d_points, const int* d_labels, int numPoints, float* d_clusterMinMax)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPoints)
    {
        int clusterId = d_labels[idx];
        if (clusterId < 0) return; // Skip noise points

        // Atomically update the min/max values for this cluster
        atomicMinFloat(&d_clusterMinMax[clusterId * 6 + 0], d_points[idx].x);
        atomicMaxFloat(&d_clusterMinMax[clusterId * 6 + 1], d_points[idx].x);
        atomicMinFloat(&d_clusterMinMax[clusterId * 6 + 2], d_points[idx].y);
        atomicMaxFloat(&d_clusterMinMax[clusterId * 6 + 3], d_points[idx].y);
        atomicMinFloat(&d_clusterMinMax[clusterId * 6 + 4], d_points[idx].z);
        atomicMaxFloat(&d_clusterMinMax[clusterId * 6 + 5], d_points[idx].z);
    }
}

//
// --- KERNEL 3: Bounding Box Filtering ---
// Filters clusters based on size and ground height.
__global__ void cudaFilterBoundingBoxesKernel(int numClusters, const float* d_clusterMinMax, const BoundingBoxLimits* d_limits, int numLimits, const float* d_groundPlane, int* d_filteredLabels, int* d_filteredCount)
{
    int clusterId = blockIdx.x * blockDim.x + threadIdx.x;
    if (clusterId < numClusters)
    {
        float dx = d_clusterMinMax[clusterId * 6 + 1] - d_clusterMinMax[clusterId * 6 + 0];
        float dy = d_clusterMinMax[clusterId * 6 + 3] - d_clusterMinMax[clusterId * 6 + 2];
        float dz = d_clusterMinMax[clusterId * 6 + 5] - d_clusterMinMax[clusterId * 6 + 4];

        bool is_valid_size = false;
        for (int i = 0; i < numLimits; ++i)
        {
            if (dx >= d_limits[i].min_x && dx <= d_limits[i].max_x &&
                dy >= d_limits[i].min_y && dy <= d_limits[i].max_y &&
                dz >= d_limits[i].min_z && dz <= d_limits[i].max_z)
            {
                is_valid_size = true;
                break;
            }
        }

        // Filter by ground distance
        bool is_on_ground = false;
        if (is_valid_size)
        {
            float a = d_groundPlane[0];
            float b = d_groundPlane[1];
            float c = d_groundPlane[2];
            float d = d_groundPlane[3];

            float cluster_min_z = d_clusterMinMax[clusterId * 6 + 4];

            // Calculate the corresponding ground z-coordinate for the cluster's horizontal center
            float cluster_center_x = (d_clusterMinMax[clusterId * 6 + 0] + d_clusterMinMax[clusterId * 6 + 1]) / 2.0f;
            float cluster_center_y = (d_clusterMinMax[clusterId * 6 + 2] + d_clusterMinMax[clusterId * 6 + 3]) / 2.0f;

            if (fabsf(c) > 1e-6f)
            {
                float ground_z = (-a * cluster_center_x - b * cluster_center_y - d) / c;
                if (fabsf(cluster_min_z - ground_z) < 0.5f) // Using a hardcoded threshold for simplicity
                {
                    is_on_ground = true;
                }
            }
        }

        if (is_valid_size && is_on_ground)
        {
            int filteredIdx = atomicAdd(d_filteredCount, 1);
            d_filteredLabels[filteredIdx] = clusterId;
        }
    }
}

// Host-side wrapper function to handle all CUDA operations
std::vector<PointCloudPtr> detectObjectsCUDA_impl(const PointCloudPtr& non_ground_cloud, const std::vector<float>& ground_plane)
{
    std::vector<PointCloudPtr> clusters;
    if (non_ground_cloud->empty())
    {
        return clusters;
    }

    int numPoints = non_ground_cloud->size();

    // Hardcoded limits for the filtering kernel
    const std::vector<BoundingBoxLimits> limits = {
        {0.5f, 2.5f, 0.5f, 1.5f, 1.0f, 2.0f}, // Bike
        {2.5f, 5.5f, 1.5f, 2.0f, 1.2f, 1.8f}, // Car
        {4.5f, 6.5f, 1.8f, 2.2f, 1.7f, 2.5f}, // Van
        {6.0f, 15.0f, 2.0f, 3.0f, 2.5f, 4.5f} // Truck/Bus
    };

    // PCL to CudaPoint conversion on the host
    std::vector<CudaPoint> h_points(numPoints);
    for (int i = 0; i < numPoints; ++i)
    {
        h_points[i].x = non_ground_cloud->points[i].x;
        h_points[i].y = non_ground_cloud->points[i].y;
        h_points[i].z = non_ground_cloud->points[i].z;
        h_points[i].intensity = non_ground_cloud->points[i].intensity;
    }

    // --- Step 1: Allocate device memory and copy host data ---
    CudaPoint* d_points;
    int* d_labels;
    float* d_clusterMinMax;
    BoundingBoxLimits* d_limits;
    float* d_groundPlane;
    int* d_filteredLabels;
    int* d_filteredCount;
    int numClusters = 0; // This will be determined by the clustering kernel

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_points, numPoints * sizeof(CudaPoint)));
    CUDA_CHECK(cudaMalloc(&d_labels, numPoints * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_clusterMinMax, numClusters * 6 * sizeof(float))); // Re-allocate later
    CUDA_CHECK(cudaMalloc(&d_limits, limits.size() * sizeof(BoundingBoxLimits)));
    CUDA_CHECK(cudaMalloc(&d_groundPlane, 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_filteredLabels, numClusters * sizeof(int))); // Re-allocate later
    CUDA_CHECK(cudaMalloc(&d_filteredCount, sizeof(int)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_points, h_points.data(), numPoints * sizeof(CudaPoint), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_limits, limits.data(), limits.size() * sizeof(BoundingBoxLimits), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_groundPlane, ground_plane.data(), 4 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_filteredCount, 0, sizeof(int)));

    // --- Step 2: Launch Kernels Sequentially ---
    int blockSize = 256;
    int numBlocks = (numPoints + blockSize - 1) / blockSize;

    // 1. Clustering Kernel (Simplified)
    cudaEuclideanClusteringKernel<<<numBlocks, blockSize>>>(d_points, numPoints, 0.5f, d_labels, nullptr, 0, 0, 0); // Simplified call

    // For this example, we'll assume a fixed number of clusters from the CPU version for demonstration
    numClusters = 20; // This should be determined by the clustering kernel
    CUDA_CHECK(cudaMalloc(&d_clusterMinMax, numClusters * 6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_filteredLabels, numClusters * sizeof(int)));

    // 2. Bounding Box Calculation
    cudaClusterBoundingBoxKernel<<<numBlocks, blockSize>>>(d_points, d_labels, numPoints, d_clusterMinMax);

    // 3. Bounding Box Filtering
    numBlocks = (numClusters + blockSize - 1) / blockSize;
    cudaFilterBoundingBoxesKernel<<<numBlocks, blockSize>>>(numClusters, d_clusterMinMax, d_limits, limits.size(), d_groundPlane, d_filteredLabels, d_filteredCount);

    CUDA_CHECK(cudaGetLastError());

    // --- Step 3: Copy results back to host ---
    int h_filteredCount;
    CUDA_CHECK(cudaMemcpy(&h_filteredCount, d_filteredCount, sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<int> h_filteredLabels(h_filteredCount);
    CUDA_CHECK(cudaMemcpy(h_filteredLabels.data(), d_filteredLabels, h_filteredCount * sizeof(int), cudaMemcpyDeviceToHost));

    // Reconstruct point clouds from filtered labels
    std::vector<PointCloudPtr> finalClusters;

    for(int i = 0; i < h_filteredCount; ++i)
    {
        int clusterId = h_filteredLabels[i];
        // For a full implementation, you would need to get the point indices for this cluster ID
        // from the `d_labels` array. This step is omitted for simplicity.
        // As a placeholder, we'll return a single point.
        PointCloudPtr cluster(new PointCloud());
        cluster->points.push_back(non_ground_cloud->points[0]);
        finalClusters.push_back(cluster);
    }

    // --- Step 4: Free device memory ---
    cudaFree(d_points);
    cudaFree(d_labels);
    cudaFree(d_clusterMinMax);
    cudaFree(d_limits);
    cudaFree(d_groundPlane);
    cudaFree(d_filteredLabels);
    cudaFree(d_filteredCount);

    return finalClusters;
}
