/*
 * Pointcloud Clustering Detector Cuda
 * Copyright (c) 2025 Menghao Woods
 *
 * Licensed under the MIT License. See LICENSE file in the project root for details.
 */
#include "constant.h"
#include <pcl/ModelCoefficients.h>
#include <pcl/common/common.h>
#include <vector>
#include <memory>
#include <iostream>
#include <map>

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

// --- KERNEL 1: Initialize Grid ---
// Sets all grid cells to an invalid value (-1).
__global__ void cudaGridInitKernel(int* d_gridMap, int gridSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < gridSize)
    {
        d_gridMap[idx] = -1;
    }
}

// --- KERNEL 2: Build Grid ---
// Populates the grid with point indices. Assumes one point per cell for simplicity.
__global__ void cudaGridBuildKernel(const CudaPoint* d_points, int numPoints, float clusterTolerance, int* d_gridMap, float minX, float minY, float minZ, int gridSizeX, int gridSizeY)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPoints)
    {
        int gridX = static_cast<int>((d_points[idx].x - minX) / clusterTolerance);
        int gridY = static_cast<int>((d_points[idx].y - minY) / clusterTolerance);
        int gridZ = static_cast<int>((d_points[idx].z - minZ) / clusterTolerance);

        int gridIdx = gridX + gridY * gridSizeX + gridZ * gridSizeX * gridSizeY;
        atomicExch(&d_gridMap[gridIdx], idx);
    }
}

// --- KERNEL 3: Euclidean Clustering ---
// A functional Euclidean clustering kernel using a grid-based approach.
__global__ void cudaEuclideanClusteringKernel(const CudaPoint* d_points, int numPoints, float clusterTolerance, int* d_labels, const int* d_gridMap, float minX, float minY, float minZ, int gridSizeX, int gridSizeY, int gridSizeZ)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints)
    {
        return;
    }

    // Step 1: Initialize each point as its own cluster
    d_labels[idx] = idx;
    __syncthreads();

    // Step 2: Iteratively merge clusters. This loop continues until no more points are reassigned.
    bool changed = true;
    while (changed)
    {
        changed = false;
        __syncthreads();

        int myClusterId = d_labels[idx];

        // Find grid cell coordinates for the current point
        int gridX = static_cast<int>((d_points[idx].x - minX) / clusterTolerance);
        int gridY = static_cast<int>((d_points[idx].y - minY) / clusterTolerance);
        int gridZ = static_cast<int>((d_points[idx].z - minZ) / clusterTolerance);

        // Step 3: Check neighbors in the 26 surrounding grid cells
        for (int z = gridZ - 1; z <= gridZ + 1; ++z)
        {
            if (z < 0 || z >= gridSizeZ) continue;
            for (int y = gridY - 1; y <= gridY + 1; ++y)
            {
                if (y < 0 || y >= gridSizeY) continue;
                for (int x = gridX - 1; x <= gridX + 1; ++x)
                {
                    if (x < 0 || x >= gridSizeX) continue;

                    int neighborGridIdx = x + y * gridSizeX + z * gridSizeX * gridSizeY;
                    int neighborPointIdx = d_gridMap[neighborGridIdx];

                    if (neighborPointIdx != -1)
                    {
                        float distSq = (d_points[idx].x - d_points[neighborPointIdx].x) * (d_points[idx].x - d_points[neighborPointIdx].x) +
                                       (d_points[idx].y - d_points[neighborPointIdx].y) * (d_points[idx].y - d_points[neighborPointIdx].y) +
                                       (d_points[idx].z - d_points[neighborPointIdx].z) * (d_points[idx].z - d_points[neighborPointIdx].z);

                        if (distSq < clusterTolerance * clusterTolerance)
                        {
                            int neighborClusterId = d_labels[neighborPointIdx];
                            if (myClusterId < neighborClusterId)
                            {
                                if (atomicCAS(&d_labels[neighborPointIdx], neighborClusterId, myClusterId) == neighborClusterId)
                                {
                                    changed = true;
                                }
                            }
                            else if (neighborClusterId < myClusterId)
                            {
                                if (atomicCAS(&d_labels[idx], myClusterId, neighborClusterId) == myClusterId)
                                {
                                    changed = true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

//
// --- KERNEL 4: Bounding Box Calculation ---
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
// --- KERNEL 5: Bounding Box Filtering ---
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

    // Get cloud's bounding box for grid calculation
    Point min_pt, max_pt;
    pcl::getMinMax3D(*non_ground_cloud, min_pt, max_pt);

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
    int* d_gridMap;

    // Allocate device memory for points, labels, and ground plane
    CUDA_CHECK(cudaMalloc(&d_points, numPoints * sizeof(CudaPoint)));
    CUDA_CHECK(cudaMalloc(&d_labels, numPoints * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_limits, limits.size() * sizeof(BoundingBoxLimits)));
    CUDA_CHECK(cudaMalloc(&d_groundPlane, 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_filteredCount, sizeof(int)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_points, h_points.data(), numPoints * sizeof(CudaPoint), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_limits, limits.data(), limits.size() * sizeof(BoundingBoxLimits), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_groundPlane, ground_plane.data(), 4 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_filteredCount, 0, sizeof(int)));

    // --- Step 2: Grid Creation and Clustering ---
    // Get cluster tolerance from config, or hardcode here.
    float clusterTolerance = 0.5f;

    // Calculate grid dimensions
    int gridSizeX = static_cast<int>((max_pt.x - min_pt.x) / clusterTolerance) + 1;
    int gridSizeY = static_cast<int>((max_pt.y - min_pt.y) / clusterTolerance) + 1;
    int gridSizeZ = static_cast<int>((max_pt.z - min_pt.z) / clusterTolerance) + 1;
    int gridSize = gridSizeX * gridSizeY * gridSizeZ;

    // Allocate and initialize the grid map
    CUDA_CHECK(cudaMalloc(&d_gridMap, gridSize * sizeof(int)));
    int blockSize = 256;
    int numBlocks = (gridSize + blockSize - 1) / blockSize;
    cudaGridInitKernel<<<numBlocks, blockSize>>>(d_gridMap, gridSize);

    // Build the grid map
    numBlocks = (numPoints + blockSize - 1) / blockSize;
    cudaGridBuildKernel<<<numBlocks, blockSize>>>(d_points, numPoints, clusterTolerance, d_gridMap, min_pt.x, min_pt.y, min_pt.z, gridSizeX, gridSizeY);

    // Launch the main clustering kernel
    cudaEuclideanClusteringKernel<<<numBlocks, blockSize>>>(d_points, numPoints, clusterTolerance, d_labels, d_gridMap, min_pt.x, min_pt.y, min_pt.z, gridSizeX, gridSizeY, gridSizeZ);

    // --- Step 3: Find unique clusters and prepare for filtering ---
    std::vector<int> h_labels(numPoints);
    CUDA_CHECK(cudaMemcpy(h_labels.data(), d_labels, numPoints * sizeof(int), cudaMemcpyDeviceToHost));

    std::map<int, int> cluster_map;
    int uniqueClusterId = 0;
    for(int i = 0; i < numPoints; ++i)
    {
        if (cluster_map.find(h_labels[i]) == cluster_map.end())
        {
            cluster_map[h_labels[i]] = uniqueClusterId++;
        }
    }
    int numClusters = uniqueClusterId;

    // Remap labels on host (for correct indexing on GPU)
    for(int i = 0; i < numPoints; ++i)
    {
        h_labels[i] = cluster_map[h_labels[i]];
    }

    // Copy remapped labels back to device
    CUDA_CHECK(cudaMemcpy(d_labels, h_labels.data(), numPoints * sizeof(int), cudaMemcpyHostToDevice));

    // Allocate memory for min/max values and filtered labels
    CUDA_CHECK(cudaMalloc(&d_clusterMinMax, numClusters * 6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_filteredLabels, numClusters * sizeof(int)));

    // Initialize min/max values
    std::vector<float> h_initial_minmax(numClusters * 6);
    for (int i = 0; i < numClusters; ++i)
    {
        h_initial_minmax[i * 6 + 0] = 1e9f; // xmin
        h_initial_minmax[i * 6 + 1] = -1e9f; // xmax
        h_initial_minmax[i * 6 + 2] = 1e9f;  // ymin
        h_initial_minmax[i * 6 + 3] = -1e9f; // ymax
        h_initial_minmax[i * 6 + 4] = 1e9f;  // zmin
        h_initial_minmax[i * 6 + 5] = -1e9f; // zmax
    }
    CUDA_CHECK(cudaMemcpy(d_clusterMinMax, h_initial_minmax.data(), numClusters * 6 * sizeof(float), cudaMemcpyHostToDevice));

    // 4. Bounding Box Calculation
    numBlocks = (numPoints + blockSize - 1) / blockSize;
    cudaClusterBoundingBoxKernel<<<numBlocks, blockSize>>>(d_points, d_labels, numPoints, d_clusterMinMax);

    // 5. Bounding Box Filtering
    numBlocks = (numClusters + blockSize - 1) / blockSize;
    cudaFilterBoundingBoxesKernel<<<numBlocks, blockSize>>>(numClusters, d_clusterMinMax, d_limits, limits.size(), d_groundPlane, d_filteredLabels, d_filteredCount);

    CUDA_CHECK(cudaGetLastError());

    // --- Step 4: Copy results back to host ---
    int h_filteredCount;
    CUDA_CHECK(cudaMemcpy(&h_filteredCount, d_filteredCount, sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<int> h_filteredLabels(h_filteredCount);
    CUDA_CHECK(cudaMemcpy(h_filteredLabels.data(), d_filteredLabels, h_filteredCount * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<int> h_labels_final(numPoints);
    CUDA_CHECK(cudaMemcpy(h_labels_final.data(), d_labels, numPoints * sizeof(int), cudaMemcpyDeviceToHost));

    // Reconstruct point clouds from filtered labels
    std::vector<PointCloudPtr> finalClusters;
    finalClusters.reserve(h_filteredCount);

    std::map<int, PointCloudPtr> clusters_map;
    for(int i = 0; i < h_filteredCount; ++i)
    {
        clusters_map[h_filteredLabels[i]] = PointCloudPtr(new PointCloud());
    }

    for(int i = 0; i < numPoints; ++i)
    {
        int clusterId = h_labels_final[i];
        if (clusters_map.count(clusterId))
        {
            clusters_map[clusterId]->points.push_back(non_ground_cloud->points[i]);
        }
    }

    for (auto const& [id, cloud] : clusters_map)
    {
        cloud->width = cloud->points.size();
        cloud->height = 1;
        cloud->is_dense = true;
        finalClusters.push_back(cloud);
    }

    // --- Step 5: Free device memory ---
    cudaFree(d_points);
    cudaFree(d_labels);
    cudaFree(d_clusterMinMax);
    cudaFree(d_limits);
    cudaFree(d_groundPlane);
    cudaFree(d_filteredLabels);
    cudaFree(d_filteredCount);
    cudaFree(d_gridMap);

    return finalClusters;
}
