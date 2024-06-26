// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is an implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

#ifndef LOAM_LASERMAPPING_H
#define LOAM_LASERMAPPING_H

#include "BasicLaserMapping.h"
#include "common.h"

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

namespace loam
{
   /** \briefLOAM  激光测绘组件的实施。
    * `LaserMapping`类是LOAM算法核心实现的一部分，它通过ROS框架订阅多种传感器数据，利用这些数据进行激光雷达的定位与三维地图构建。该类的设计允许灵活地处理来自不同传感器的数据流，实现高效的在线SLAM功能，对于自动驾驶、机器人导航等领域具有重要应用价值
    */
   class LaserMapping : public BasicLaserMapping
   {
   public:
      explicit LaserMapping(const float &scanPeriod = 0.1, const size_t &maxIterations = 10);

      /** \brief 在活动模式下设置组件。
       *
       * @param node the ROS node handle
       * @param privateNode 专用ROS节点句柄
       */
      virtual bool setup(ros::NodeHandle &node, ros::NodeHandle &privateNode);

      /** \brief Handler method for a new last corner cloud.
       *
       * @param cornerPointsLastMsg the new last corner cloud message
       * 处理接收到的角点云
       */
      void laserCloudCornerLastHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsLastMsg);

      /** \brief Handler method for a new last surface cloud.
       *
       * @param surfacePointsLastMsg the new last surface cloud message
       * 处理接收表面点云
       */
      void laserCloudSurfLastHandler(const sensor_msgs::PointCloud2ConstPtr &surfacePointsLastMsg);

      /** \brief Handler method for a new full resolution cloud.
       *
       * @param laserCloudFullResMsg the new full resolution cloud message
       * 处理接收全分辨率点云
       */
      void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullResMsg);

      /** \brief Handler method for a new laser odometry.
       *
       * @param laserOdometry the new laser odometry message
       * 处理接收激光里程计
       */
      void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &laserOdometry);

      /** \brief Handler method for IMU messages.
       *
       * @param imuIn the new IMU message
       * 处理接收IMU数据的消息回调函数
       */
      void imuHandler(const sensor_msgs::Imu::ConstPtr &imuIn);

      /** \brief Process incoming messages in a loop until shutdown (used in active mode). */
      void spin();

      /** \brief Try to process buffered data. */
      void process();

   protected:
      /** \brief Reset flags, etc. */
      void reset();

      /** \brief Check if all required information for a new processing step is available. */
      bool hasNewData();

      /** \brief Publish the current result via the respective topics. */
      void publishResult();

   private:
      ros::Time _timeLaserCloudCornerLast; ///< time of current last corner cloud
      ros::Time _timeLaserCloudSurfLast;   ///< time of current last surface cloud
      ros::Time _timeLaserCloudFullRes;    ///< time of current full resolution cloud
      ros::Time _timeLaserOdometry;        ///< time of current laser odometry

      bool _newLaserCloudCornerLast; ///< flag if a new last corner cloud has been received
      bool _newLaserCloudSurfLast;   ///< flag if a new last surface cloud has been received
      bool _newLaserCloudFullRes;    ///< flag if a new full resolution cloud has been received
      bool _newLaserOdometry;        ///< flag if a new laser odometry has been received

      nav_msgs::Odometry _odomAftMapped;    ///< mapping odometry message
      tf::StampedTransform _aftMappedTrans; ///< mapping odometry transformation

      ros::Publisher _pubLaserCloudSurround;   ///< map cloud message publisher
      ros::Publisher _pubLaserCloudFullRes;    ///< current full resolution cloud message publisher
      ros::Publisher _pubOdomAftMapped;        ///< mapping odometry publisher
      tf::TransformBroadcaster _tfBroadcaster; ///< mapping odometry transform broadcaster

      ros::Subscriber _subLaserCloudCornerLast; ///< last corner cloud message subscriber
      ros::Subscriber _subLaserCloudSurfLast;   ///< last surface cloud message subscriber
      ros::Subscriber _subLaserCloudFullRes;    ///< full resolution cloud message subscriber
      ros::Subscriber _subLaserOdometry;        ///< laser odometry message subscriber
      ros::Subscriber _subImu;                  ///< IMU message subscriber
   };

} // end namespace loam

#endif // LOAM_LASERMAPPING_H
