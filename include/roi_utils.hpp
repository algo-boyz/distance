#pragma once

#include "config.hpp"
#include <opencv2/imgproc.hpp>
#include <optional>

namespace vdm {

// Returns true if the point lies inside the given polygon (including boundary)
bool isPointInRoi(const cv::Point2f& point, const std::vector<cv::Point>& polygon);

// Classify a vehicle center into one of the three ROI zones
Zone getVehicleRoiZone(const cv::Point2f& center);

// Scale ROI polygons from the original calibration resolution to the actual frame size
std::map<Zone, std::vector<cv::Point>> scaleRoiZones(int frameWidth, int frameHeight);

} // namespace vdm
