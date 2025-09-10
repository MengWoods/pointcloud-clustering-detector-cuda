rm -rf build
mkdir build && cd build
cmake ..
make -j$(nproc)
./pointcloud_clustering_detector_cuda /home/mh/Documents/github/pointcloud-clustering-detector-cuda/pointcloud_clustering_detector_cuda/config/detector.yaml
cd ..
