
3. **LOAM (Laser SLAM)**:
   - **简介**: LOAM（Lidar Odometry and Mapping）是一种激光雷达SLAM方法，特别适合于三维环境的实时定位和建图。它由Ji Zhang等人提出，针对旋转激光雷达（如Velodyne）设计。
   - **特点**: 通过将问题分解为局部里程计（odometry）和局部映射两个部分，并在两个模块间交替进行，以减少计算负担，提高实时性能。LOAM擅长处理高动态环境中的大范围场景。
   - 


# loam_velodyne

![Screenshot](/capture.bmp)
Sample map built from [nsh_indoor_outdoor.bag](http://www.frc.ri.cmu.edu/~jizhang03/Datasets/nsh_indoor_outdoor.bag) (opened with [ccViewer](http://www.danielgm.net/cc/))

:white_check_mark: Tested with ROS Indigo and Velodyne VLP16. [(Screencast)](https://youtu.be/o1cLXY-Es54)

All sources were taken from [ROS documentation](http://docs.ros.org/indigo/api/loam_velodyne/html/files.html)

Ask questions [here](https://github.com/laboshinl/loam_velodyne/issues/3).

## How to build with catkin

```
$ cd ~/catkin_ws/src/
$ git clone https://github.com/laboshinl/loam_velodyne.git
$ cd ~/catkin_ws
$ catkin_make -DCMAKE_BUILD_TYPE=Release
$ source ~/catkin_ws/devel/setup.bash
```

## Running

```
roslaunch loam_velodyne loam_velodyne.launch
```

In second terminal play sample velodyne data from [VLP16 rosbag](http://www.frc.ri.cmu.edu/~jizhang03/Datasets/):
```
rosbag play ~/Downloads/velodyne.bag 
```

Or read from velodyne [VLP16 sample pcap](https://midas3.kitware.com/midas/folder/12979):
```
roslaunch velodyne_pointcloud VLP16_points.launch pcap:="$HOME/Downloads/velodyne.pcap"
```

## Troubleshooting

### `multiScanRegistration` crashes right after playing bag file

Issues [#71](https://github.com/laboshinl/loam_velodyne/issues/71) and
[#7](https://github.com/laboshinl/loam_velodyne/issues/7) address this
problem. The current known solution is to build the same version of PCL that
you have on your system from source, and set the `CMAKE_PREFIX_PATH`
accordingly so that catkin can find it. See [this
issue](https://github.com/laboshinl/loam_velodyne/issues/71#issuecomment-416024816)
for more details.


---
[Quantifying Aerial LiDAR Accuracy of LOAM for Civil Engineering Applications.](https://ceen.et.byu.edu/sites/default/files/snrprojects/wolfe_derek.pdf) Derek Anthony Wolfe

[ROS & Loam_velodyne](https://ishiguro440.wordpress.com/2016/04/05/%E5%82%99%E5%BF%98%E9%8C%B2%E3%80%80ros-loam_velodyne/) 
