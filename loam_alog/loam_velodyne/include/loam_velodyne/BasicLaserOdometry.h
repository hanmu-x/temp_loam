#pragma once
#include "Twist.h"
#include "nanoflann_pcl.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace loam
{

  /** \brief Implementation of the LOAM laser odometry component
   *
   */
  class BasicLaserOdometry
  {
  public:
    // 接受激光扫描周期和最大迭代次数作为参数
    explicit BasicLaserOdometry(float scanPeriod = 0.1, size_t maxIterations = 25);

    /** \brief 尝试处理缓冲区中的数据，执行点云匹配和位姿优化 */
    void process();
    // 更新IMU数据，传入IMU坐标变换的点云表示
    void updateIMU(pcl::PointCloud<pcl::PointXYZ> const &imuTrans);

    auto &cornerPointsSharp() { return _cornerPointsSharp; }
    auto &cornerPointsLessSharp() { return _cornerPointsLessSharp; }
    auto &surfPointsFlat() { return _surfPointsFlat; }
    auto &surfPointsLessFlat() { return _surfPointsLessFlat; }
    auto &laserCloud() { return _laserCloud; }

    auto const &transformSum() { return _transformSum; }
    auto const &transform() { return _transform; }
    auto const &lastCornerCloud() { return _lastCornerCloud; }
    auto const &lastSurfaceCloud() { return _lastSurfaceCloud; }

    void setScanPeriod(float val) { _scanPeriod = val; }
    void setMaxIterations(size_t val) { _maxIterations = val; }
    void setDeltaTAbort(float val) { _deltaTAbort = val; }
    void setDeltaRAbort(float val) { _deltaRAbort = val; }

    auto frameCount() const { return _frameCount; }
    auto scanPeriod() const { return _scanPeriod; }
    auto maxIterations() const { return _maxIterations; }
    auto deltaTAbort() const { return _deltaTAbort; }
    auto deltaRAbort() const { return _deltaRAbort; }

    /** \brief 将给定点云变换到扫描周期结束时的坐标系下
     * 将输入的点云 (cloud) 中的每个点，从其采集时刻的局部坐标系变换到整个扫描周期结束时的全局坐标系中
     * 对输入点云中的每个点应用一系列复杂的逆向位姿变换和IMU数据融合，实现了将点云数据从其采集时的局部坐标系转换到整个扫描周期结束时的全局坐标系，这对于后续的里程计计算和地图构建至关重要
     * @param 将给定点云变换到扫描周期结束时的坐标系下
     */
    size_t transformToEnd(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud);

  private:
    /** \brief 将单个点从当前坐标变换回扫描起始时的坐标系
     *
     * @param pi pi要变换的点
     * @param po 用于存储结果的点实例
     */
    void transformToStart(const pcl::PointXYZI &pi, pcl::PointXYZI &po);

    // 基于给定的一系列角度（表示不同坐标系之间的旋转）计算一个新的旋转角度，以反映整体的旋转效果。函数接收9个输入参数（分别代表三个连续旋转的欧拉角，每个旋转由绕x轴(bcx, bcy, bcz)，绕y轴(blx, bly, blz)，以及绕z轴(alx, aly, alz)的旋转角度组成），并计算出最终旋转后的三个欧拉角 (acx, acy, acz)。这种计算常用于航位推算或传感器数据融合场景，特别是当需要将多个旋转组合在一起时
    void pluginIMURotation(const Angle &bcx, const Angle &bcy, const Angle &bcz,
                           const Angle &blx, const Angle &bly, const Angle &blz,
                           const Angle &alx, const Angle &aly, const Angle &alz,
                           Angle &acx, Angle &acy, Angle &acz);

    // 累加旋转角度
    void accumulateRotation(Angle cx, Angle cy, Angle cz,
                            Angle lx, Angle ly, Angle lz,
                            Angle &ox, Angle &oy, Angle &oz);

  private:
    float _scanPeriod;     ///< 激光扫描周期
    long _frameCount;      ///< 已处理的帧数
    size_t _maxIterations; ///< 优化的最大迭代次数
    bool _systemInited;    ///< 系统是否已经初始化的标志

    float _deltaTAbort; ///< 优化过程中的时间差中止阈值
    float _deltaRAbort; ///< 优化过程中的角度差中止阈值

    pcl::PointCloud<pcl::PointXYZI>::Ptr _lastCornerCloud;  ///< 上一帧角点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr _lastSurfaceCloud; ///< 上一帧表面点云

    pcl::PointCloud<pcl::PointXYZI>::Ptr _laserCloudOri; ///< 未经处理的原始点云数据
    pcl::PointCloud<pcl::PointXYZI>::Ptr _coeffSel;      ///< 未经处理的原始点云对应的优化或筛选系数

    nanoflann::KdTreeFLANN<pcl::PointXYZI> _lastCornerKDTree;  ///< 上一帧角点KD树
    nanoflann::KdTreeFLANN<pcl::PointXYZI> _lastSurfaceKDTree; ///< 上一帧表面点KD树

    // **尖锐角点云**, **较钝角点云**, **平坦表面点云**, **非平坦表面点云** - 分类存储不同特性的点云，根据点的几何特性（如曲率）进行区分，这些分类有助于后续的特征匹配和优化
    pcl::PointCloud<pcl::PointXYZI>::Ptr _cornerPointsSharp;     ///< 尖锐角点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr _cornerPointsLessSharp; ///< 较钝角点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr _surfPointsFlat;        ///< 平坦表面点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr _surfPointsLessFlat;    ///< 非平坦表面点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr _laserCloud;            ///< 全分辨率云

    // 多组索引缓冲，用于存储在K-D树搜索过程中找到的与当前点相邻的点的索引，这些索引用于计算点之间的几何关系和优化位姿
    std::vector<int> _pointSearchCornerInd1; ///< first corner point search index buffer
    std::vector<int> _pointSearchCornerInd2; ///< second corner point search index buffer

    std::vector<int> _pointSearchSurfInd1; ///< first surface point search index buffer
    std::vector<int> _pointSearchSurfInd2; ///< second surface point search index buffer
    std::vector<int> _pointSearchSurfInd3; ///< third surface point search index buffer

    //  **当前变换** 和 **累计变换** - 分别存储本次扫描帧相对于前一帧的位姿变换和从系统启动至今的累计位姿变换，是核心的输出数据
    Twist _transform;    ///< optimized pose transformation
    Twist _transformSum; ///< accumulated optimized pose transformation

    // 分别记录每次扫描开始和结束时刻IMU测量到的roll（翻滚）、pitch（俯仰）和yaw（偏航）角
    Angle _imuRollStart, _imuPitchStart, _imuYawStart;
    Angle _imuRollEnd, _imuPitchEnd, _imuYawEnd;

    //  **IMU从起点的位移** 和 **IMU从起点的速度**
    Vector3 _imuShiftFromStart;
    Vector3 _imuVeloFromStart;
  };

} // end namespace loam
