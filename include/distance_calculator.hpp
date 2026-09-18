#pragma once

#include "config.hpp"
#include <opencv2/core.hpp>

namespace vdm {

struct Detection {
    cv::Rect2f bbox;
    int        classId   = -1;
    float      confidence = 0.f;
    Zone       zone      = Zone::NONE;
    float      distance  = 0.f;
};

// Compute monocular distance with perspective correction
// d = (h_real * f) / h_image * (1 + alpha * delta)
float calculateDistance(const cv::Rect2f& bbox, int classId, Zone zone);

// Whether the distance label should be drawn for this zone
bool shouldDisplayDistance(float distance, Zone zone);

// Color for the distance label (BGR)
cv::Scalar getDistanceColor(float distance, Zone zone);

} // namespace vdm
