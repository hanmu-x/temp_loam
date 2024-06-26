#ifndef LOAM_TWIST_H
#define LOAM_TWIST_H

#include "Angle.h"
#include "Vector3.h"

namespace loam
{

  /** \brief Twist 类包含了四个成员变量 rot_x、rot_y、rot_z 和 pos，分别表示绕 x、y、z 轴的旋转角度和位置
   * 用于处理机器人的姿态、位置和旋转信息
   */
  class Twist
  {
  public:
    Twist()
        : rot_x(),
          rot_y(),
          rot_z(),
          pos(){};

    Angle rot_x;
    Angle rot_y;
    Angle rot_z;
    Vector3 pos;
  };

} // end namespace loam

#endif // LOAM_TWIST_H
