#pragma once

#include <utility>
#include <vector>

#include <pcl/point_cloud.h>

#include "Angle.h"
#include "Vector3.h"
#include "CircularBuffer.h"
#include "time_utils.h"

namespace loam
{

  /** \brief A pair describing the start end end index of a range. */
  typedef std::pair<size_t, size_t> IndexRange;

  /** Point label options. */
  enum PointLabel
  {
    CORNER_SHARP = 2,      ///< 锐角点
    CORNER_LESS_SHARP = 1, ///< 较钝的角点
    SURFACE_LESS_FLAT = 0, ///< 较不平坦表面点
    SURFACE_FLAT = -1      ///< 平坦表面点
  };

  /** Scan Registration configuration parameters. */
  class RegistrationParams
  {
  public:
    RegistrationParams(const float &scanPeriod_ = 0.1,
                       const int &imuHistorySize_ = 200,
                       const int &nFeatureRegions_ = 6,
                       const int &curvatureRegion_ = 5,
                       const int &maxCornerSharp_ = 2,
                       const int &maxSurfaceFlat_ = 4,
                       const float &lessFlatFilterSize_ = 0.2,
                       const float &surfaceCurvatureThreshold_ = 0.1);

    
    float scanPeriod; /** 每次扫描的时间 */
    int imuHistorySize; /** IMU 历史记录状态缓冲区的大小. */
    int nFeatureRegions; /** 用于在扫描中分发特征提取的（大小相等的）区域数. */
    int curvatureRegion; /** 用于计算点曲率的周围点数（+ 点周围的区域） */
    int maxCornerSharp;  /** 每个要素区域的最大尖角点数 */
    int maxCornerLessSharp; /** 每个要素区域中不太尖锐的角点的最大数量 */
    int maxSurfaceFlat; /** 每个要素区域的最大平面点数 */
    float lessFlatFilterSize; /** 用于缩小其余不太平坦曲面点的体素大小 */
    float surfaceCurvatureThreshold; /** 低于某个点上方的曲率阈值被视为平坦的拐角点 */
  };

  /** IMU state data. */
  typedef struct IMUState
  {
    Time stamp; /** 导致此状态的测量时间（以秒为单位） */
    Angle roll; /** 当前滚动角度 */
    Angle pitch;  /** 当前俯仰角 */
    Angle yaw;  /** 当前偏航角 */
    Vector3 position; /** 3D空间中累积的全局IMU位置 */
    Vector3 velocity; /**3D空间中累积的全局IMU速度 */
    Vector3 acceleration; /** 3D空间中的当前（局部）IMU加速度. */
    
    /** \brief 在两个 IMU 状态之间插值
     *
     * @param start the first IMUState
     * @param end the second IMUState
     * @param ratio the interpolation ratio
     * @param result the target IMUState for storing the interpolation result
     */
    static void interpolate(const IMUState &start,
                            const IMUState &end,
                            const float &ratio,
                            IMUState &result)
    {
      float invRatio = 1 - ratio;

      result.roll = start.roll.rad() * invRatio + end.roll.rad() * ratio;
      result.pitch = start.pitch.rad() * invRatio + end.pitch.rad() * ratio;
      if (start.yaw.rad() - end.yaw.rad() > M_PI)
      {
        result.yaw = start.yaw.rad() * invRatio + (end.yaw.rad() + 2 * M_PI) * ratio;
      }
      else if (start.yaw.rad() - end.yaw.rad() < -M_PI)
      {
        result.yaw = start.yaw.rad() * invRatio + (end.yaw.rad() - 2 * M_PI) * ratio;
      }
      else
      {
        result.yaw = start.yaw.rad() * invRatio + end.yaw.rad() * ratio;
      }

      result.velocity = start.velocity * invRatio + end.velocity * ratio;
      result.position = start.position * invRatio + end.position * ratio;
    };
  } IMUState;

  class BasicScanRegistration
  {
  public:
    /** \brief 将新云处理为一组扫描线
     *
     * @param relTime the time relative to the scan time
     */
    void processScanlines(const Time &scanTime, std::vector<pcl::PointCloud<pcl::PointXYZI>> const &laserCloudScans);

    bool configure(const RegistrationParams &config = RegistrationParams());

    /** \brief Update new IMU state. NOTE: MUTATES ARGS! */
    void updateIMUData(Vector3 &acc, IMUState &newState);

    /** \brief Project a point to the start of the sweep using corresponding IMU data
     *
     * @param point The point to modify
     * @param relTime The time to project by
     */
    void projectPointToStartOfSweep(pcl::PointXYZI &point, float relTime);

    auto const &imuTransform() { return _imuTrans; }
    auto const &sweepStart() { return _sweepStart; }
    auto const &laserCloud() { return _laserCloud; }
    auto const &cornerPointsSharp() { return _cornerPointsSharp; }
    auto const &cornerPointsLessSharp() { return _cornerPointsLessSharp; }
    auto const &surfacePointsFlat() { return _surfacePointsFlat; }
    auto const &surfacePointsLessFlat() { return _surfacePointsLessFlat; }
    auto const &config() { return _config; }

  private:
    /** \brief Check is IMU data is available. */
    inline bool hasIMUData() { return _imuHistory.size() > 0; };

    /** \brief Set up the current IMU transformation for the specified relative time.
     *
     * @param relTime the time relative to the scan time
     */
    void setIMUTransformFor(const float &relTime);

    /** \brief Project the given point to the start of the sweep, using the current IMU state and position shift.
     *
     * @param point the point to project
     */
    void transformToStartIMU(pcl::PointXYZI &point);

    /** \brief Prepare for next scan / sweep.
     *
     * @param scanTime the current scan time
     * @param newSweep indicator if a new sweep has started
     */
    void reset(const Time &scanTime);

    /** \brief Extract features from current laser cloud.
     *
     * @param beginIdx the index of the first scan to extract features from
     */
    void extractFeatures(const uint16_t &beginIdx = 0);

    /** \brief Set up region buffers for the specified point range.
     *
     * @param startIdx the region start index
     * @param endIdx the region end index
     */
    void setRegionBuffersFor(const size_t &startIdx,
                             const size_t &endIdx);

    /** \brief Set up scan buffers for the specified point range.
     *
     * @param startIdx the scan start index
     * @param endIdx the scan start index
     */
    void setScanBuffersFor(const size_t &startIdx,
                           const size_t &endIdx);

    /** \brief Mark a point and its neighbors as picked.
     *
     * This method will mark neighboring points within the curvature region as picked,
     * as long as they remain within close distance to each other.
     *
     * @param cloudIdx the index of the picked point in the full resolution cloud
     * @param scanIdx the index of the picked point relative to the current scan
     */
    void markAsPicked(const size_t &cloudIdx,
                      const size_t &scanIdx);

    /** \brief Try to interpolate the IMU state for the given time.
     *
     * @param relTime the time relative to the scan time
     * @param outputState the output state instance
     */
    void interpolateIMUStateFor(const float &relTime, IMUState &outputState);

    void updateIMUTransform();

  private:
    RegistrationParams _config; ///< registration parameter

    pcl::PointCloud<pcl::PointXYZI> _laserCloud; ///< full resolution input cloud
    std::vector<IndexRange> _scanIndices;        ///< start and end indices of the individual scans withing the full resolution cloud

    pcl::PointCloud<pcl::PointXYZI> _cornerPointsSharp;     ///< sharp corner points cloud
    pcl::PointCloud<pcl::PointXYZI> _cornerPointsLessSharp; ///< less sharp corner points cloud
    pcl::PointCloud<pcl::PointXYZI> _surfacePointsFlat;     ///< flat surface points cloud
    pcl::PointCloud<pcl::PointXYZI> _surfacePointsLessFlat; ///< less flat surface points cloud

    Time _sweepStart;                     ///< time stamp of beginning of current sweep
    Time _scanTime;                       ///< time stamp of most recent scan
    IMUState _imuStart;                   ///< the interpolated IMU state corresponding to the start time of the currently processed laser scan
    IMUState _imuCur;                     ///< the interpolated IMU state corresponding to the time of the currently processed laser scan point
    Vector3 _imuPositionShift;            ///< position shift between accumulated IMU position and interpolated IMU position
    size_t _imuIdx = 0;                   ///< the current index in the IMU history
    CircularBuffer<IMUState> _imuHistory; ///< history of IMU states for cloud registration

    pcl::PointCloud<pcl::PointXYZ> _imuTrans = {4, 1}; ///< IMU transformation information

    std::vector<float> _regionCurvature;    ///< point curvature buffer
    std::vector<PointLabel> _regionLabel;   ///< point label buffer
    std::vector<size_t> _regionSortIndices; ///< sorted region indices based on point curvature
    std::vector<int> _scanNeighborPicked;   ///< flag if neighboring point was already picked
  };

}
