#pragma once
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

#include "Twist.h"
#include "CircularBuffer.h"
#include "time_utils.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

namespace loam
{

   /** IMU state data. IMUState2 结构体用于存储IMU（惯性测量单元）的状态数据，包括时间戳、滚动角、俯仰角。 */
   typedef struct IMUState2
   {
      /** 导致此状态的测量时间（以秒为单位） */
      Time stamp;

      /** 当前滚动角度 */
      Angle roll;

      /** 当前俯仰角 */
      Angle pitch;

      /** \brief 在两个 IMU 状态之间插值
       * 用于在两个IMU状态之间进行线性插值，计算给定比例ratio处的新IMU状态
       * @param start 第一个 IMU 状态
       * @param end 第二个 IMU 状态
       * @param ratio 插值比
       * @param result 用于存储插值结果的目标 IMU 状态
       */
      static void interpolate(const IMUState2 &start,
                              const IMUState2 &end,
                              const float &ratio,
                              IMUState2 &result)
      {
         float invRatio = 1 - ratio;

         result.roll = start.roll.rad() * invRatio + end.roll.rad() * ratio;
         result.pitch = start.pitch.rad() * invRatio + end.pitch.rad() * ratio;
      };
   } IMUState2;

   // BasicLaserMapping类是LOAM算法的核心基础类，封装了激光雷达数据处理、IMU数据融合、位姿估计与地图构建等基本功能
   class BasicLaserMapping
   {
   public:
      // 可设置扫描周期和最大迭代次数等参数 
      explicit BasicLaserMapping(const float &scanPeriod = 0.1, const size_t &maxIterations = 10);

      /** \brief 尝试处理缓冲数据. */
      bool process(Time const &laserOdometryTime);
      void updateIMU(IMUState2 const &newState); // 可设置扫描周期和最大迭代次数等参数
      // 更新里程计信息，有两种重载形式，分别接受欧拉角+位置或Twist形式的输入
      void updateOdometry(double pitch, double yaw, double roll, double x, double y, double z);
      void updateOdometry(Twist const &twist);

      auto &laserCloud() { return *_laserCloudFullRes; }
      auto &laserCloudCornerLast() { return *_laserCloudCornerLast; }
      auto &laserCloudSurfLast() { return *_laserCloudSurfLast; }

      void setScanPeriod(float val) { _scanPeriod = val; }
      void setMaxIterations(size_t val) { _maxIterations = val; }
      void setDeltaTAbort(float val) { _deltaTAbort = val; }
      void setDeltaRAbort(float val) { _deltaRAbort = val; }

      auto &downSizeFilterCorner() { return _downSizeFilterCorner; }
      auto &downSizeFilterSurf() { return _downSizeFilterSurf; }
      auto &downSizeFilterMap() { return _downSizeFilterMap; }

      auto frameCount() const { return _frameCount; }
      auto scanPeriod() const { return _scanPeriod; }
      auto maxIterations() const { return _maxIterations; }
      auto deltaTAbort() const { return _deltaTAbort; }
      auto deltaRAbort() const { return _deltaRAbort; }

      auto const &transformAftMapped() const { return _transformAftMapped; }
      auto const &transformBefMapped() const { return _transformBefMapped; }
      auto const &laserCloudSurroundDS() const { return *_laserCloudSurroundDS; }

      bool hasFreshMap() const { return _downsizedMapCreated; }

   private:
      /** 优化待映射变换，核心算法之一，用于优化位姿估计 */
      void optimizeTransformTobeMapped();

      void transformAssociateToMap();
      void transformUpdate();
      void pointAssociateToMap(const pcl::PointXYZI &pi, pcl::PointXYZI &po);
      void pointAssociateTobeMapped(const pcl::PointXYZI &pi, pcl::PointXYZI &po);
      void transformFullResToMap();

      bool createDownsizedMap();

      // private:
      size_t toIndex(int i, int j, int k) const
      {
         return i + _laserCloudWidth * j + _laserCloudWidth * _laserCloudHeight * k;
      }

   private:
      Time _laserOdometryTime; // 存储上一次激光里程计处理的时间戳，用于跟踪数据处理的时间顺序

      float _scanPeriod; // 扫描周期，单位为秒，表示每次激光雷达完成一次扫描所需的时间
      const int _stackFrameNum; // 分别代表角点云和表面点云的帧堆栈数量，用于存储历史数据以辅助地图构建和位姿估计。
      const int _mapFrameNum;
      long _frameCount; // 当前处理的帧数和地图帧数，用于跟踪处理进度。
      long _mapFrameCount; // 优化的最大迭代次数，控制位姿估计过程中的迭代深度

      size_t _maxIterations; ///< 最大迭代次数
      float _deltaTAbort;    // 优化中止的阈值，当两次迭代间的位移变化（时间或角度）小于这些值时，优化提前终止
      float _deltaRAbort;    
      // 定义了点云数据的维度和总数，用于组织和管理点云数据的内存布局。
      int _laserCloudCenWidth;
      int _laserCloudCenHeight;
      int _laserCloudCenDepth;
      // 定义了完整点云数据的维度和总数，用于组织和管理点云数据的内存布局
      const size_t _laserCloudWidth;
      const size_t _laserCloudHeight;
      const size_t _laserCloudDepth;
      const size_t _laserCloudNum;
      // 存储最后角点云、表面点云和完整分辨率点云的指针
      pcl::PointCloud<pcl::PointXYZI>::Ptr _laserCloudCornerLast; 
      pcl::PointCloud<pcl::PointXYZI>::Ptr _laserCloudSurfLast;   
      pcl::PointCloud<pcl::PointXYZI>::Ptr _laserCloudFullRes;    
      // 定义了角点云和表面点云的堆栈（用于存储历史数据）及其下采样后的版本的指针
      pcl::PointCloud<pcl::PointXYZI>::Ptr _laserCloudCornerStack;
      pcl::PointCloud<pcl::PointXYZI>::Ptr _laserCloudSurfStack;
      pcl::PointCloud<pcl::PointXYZI>::Ptr _laserCloudCornerStackDS; 
      pcl::PointCloud<pcl::PointXYZI>::Ptr _laserCloudSurfStackDS;   
      // 定义了用于周围点云和地图中角点云、表面点云的指针
      pcl::PointCloud<pcl::PointXYZI>::Ptr _laserCloudSurround;
      pcl::PointCloud<pcl::PointXYZI>::Ptr _laserCloudSurroundDS; 
      pcl::PointCloud<pcl::PointXYZI>::Ptr _laserCloudCornerFromMap;
      pcl::PointCloud<pcl::PointXYZI>::Ptr _laserCloudSurfFromMap;
      // 定义了原始点云和系数选择的点云
      pcl::PointCloud<pcl::PointXYZI> _laserCloudOri;
      pcl::PointCloud<pcl::PointXYZI> _coeffSel;
      // 
      std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> _laserCloudCornerArray;
      std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> _laserCloudSurfArray;
      std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> _laserCloudCornerDSArray; 
      std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> _laserCloudSurfDSArray;
      // 
      std::vector<size_t> _laserCloudValidInd;
      std::vector<size_t> _laserCloudSurroundInd;

      Twist _transformSum, _transformIncre, _transformTobeMapped, _transformBefMapped, _transformAftMapped;

      CircularBuffer<IMUState2> _imuHistory; 

      pcl::VoxelGrid<pcl::PointXYZI> _downSizeFilterCorner; 
      pcl::VoxelGrid<pcl::PointXYZI> _downSizeFilterSurf;   
      pcl::VoxelGrid<pcl::PointXYZI> _downSizeFilterMap;    

      bool _downsizedMapCreated = false;
   };

} // end namespace loam
