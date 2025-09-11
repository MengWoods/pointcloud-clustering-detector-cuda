
# pointcloud-clustering-detector-cuda

A CUDA-accelerated point cloud clustering detector for real-time 3D perception.

<p align="center">
<img src="./resource/detector.png" width="60%">
</p>

## 🚀 Overview

This repository contains a point cloud clustering and detection system. The primary goal of this project is to accelerate the core clustering algorithm using NVIDIA's CUDA to achieve real-time performance, a crucial requirement for dense point cloud data from sources like LiDAR.

The system now offers both a **CPU-based** and a highly-optimized **CUDA-based** clustering method, selectable via the configuration file.

## ✨ Key Features

The full processing pipeline includes:

* **Ground Removal:** A robust algorithm to remove the ground plane from the point cloud.

* **Clustering Detector:** A parallel clustering algorithm to group points into objects.

* **Bounding Box Generation:** Computation of 3D bounding boxes for each detected object.

* **Smart Filtering:** A multi-stage filter that refines detections by removing clusters that are too small, too large, or not on the ground.

* **3D Visualization:** The ability to visualize the ground, non-ground points, and the detected bounding boxes in real-time.

## ⚡ Performance

A key motivation for this project was to address the performance limitations of CPU-based clustering algorithms. As shown in these two videos, the CUDA-accelerated version is significantly faster than the CPU version, making it suitable for real-time applications.

* **CPU Performance:** [[Link to CPU video](https://youtu.be/ickDlyV1Nuk)]

* **GPU Performance:** [[Link to GPU video](https://youtu.be/J843j-RZsh4)]

### **Accuracy**

While the GPU implementation delivers a massive speed increase, ongoing development is focused on improving the accuracy of the clustering and detection to reduce false positives and enhance the quality of the final output.

## 🛠️ Usage

This guide will get you up and running with the project.

**1. Configuration**
Before compiling, ensure your data paths are correctly configured. Open the `config/detector.yaml` file and update the `point_cloud_loader` section to point to your dataset.

**2. Compilation**
The project uses CMake to manage the build process. A simple bash script is provided to compile the project.

```

./compile.sh

```

This script will create a `build` directory and compile the `pointcloud_clustering_detector_cuda` executable.

**3. Switching Between CPU and GPU**
To switch between the CPU and GPU clustering methods, simply edit your `config/detector.yaml` file.

## 🗺️ Project Structure

```

├── LICENSE
├── README.md
├── pointcloud\_clustering\_detector\_cuda
│   ├── build                 (Compiled binaries and build artifacts)
│   ├── compile.sh            (Script to build the project)
│   ├── config
│   │   └── detector.yaml     (Configuration file for the detector)
│   ├── include               (Header files)
│   ├── src                   (Source files, including .cpp and .cu)
│   └── CMakeLists.txt
├── resource
│   └── detector.png

```

## ➡️ To-Do

* **Overall Speed Optimization:** Further optimize the entire pipeline for maximum throughput and minimum latency, including memory management and data loading.

* **Accuracy Improvement:** Enhance the detection filter and clustering algorithm to improve accuracy and reduce false positives.

* **Bounding Box Confidence:** Implement a confidence score for each detected bounding box to indicate the reliability of the detection.

* **Further CUDA Optimization:** Explore advanced CUDA techniques (e.g., shared memory, streams, memory coalescing) to maximize GPU utilization.
