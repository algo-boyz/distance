#include "distance_calculator.hpp"
#include <cmath>

namespace vdm {

float calculateDistance(const cv::Rect2f& bbox, int classId, Zone zone) {
    auto heightIt = REAL_VEHICLE_HEIGHTS.find(classId);
    auto centerIt = OPTICAL_CENTERS.find(zone);
    if (heightIt == REAL_VEHICLE_HEIGHTS.end() || centerIt == OPTICAL_CENTERS.end()) {
        return 0.f;
    }

    const float bboxHeight = bbox.height;
    if (bboxHeight <= 0.f) return 0.f;

    const float vehicleCx = bbox.x + bbox.width  * 0.5f;
    const float vehicleCy = bbox.y + bbox.height * 0.5f;
    const cv::Point2f& optical = centerIt->second;

    const float dx = vehicleCx - optical.x;
    const float dy = vehicleCy - optical.y;
    const float displacement = std::sqrt(dx * dx + dy * dy);

    // d = (h_real * f) / h_image
    float distance = (heightIt->second * FOCAL_LENGTH) / bboxHeight;

    // Perspective distortion correction
    distance *= (1.0f + displacement * DISPLACEMENT_COEFF);
    return distance;
}

bool shouldDisplayDistance(float distance, Zone zone) {
    if (zone == Zone::MAIN) {
        return distance <= MAX_DISPLAY_DISTANCE;
    }
    auto it = DISTANCE_DISPLAY_THRESHOLDS.find(zone);
    if (it == DISTANCE_DISPLAY_THRESHOLDS.end()) {
        return distance <= MAX_DISPLAY_DISTANCE;
    }
    return distance <= it->second;
}

cv::Scalar getDistanceColor(float distance, Zone zone) {
    // MAIN lane: red if < 5 m
    if (zone == Zone::MAIN && distance < MAIN_ROI_WARNING_THRESH) {
        return {0, 0, 255}; // BGR red
    }

    auto it = WARNING_DISTANCES.find(zone);
    const float warn = (it != WARNING_DISTANCES.end()) ? it->second : 2.0f;

    if (distance < warn) {
        return {0, 0, 255}; // red
    }
    return {0, 255, 0}; // green
}

} // namespace vdm
