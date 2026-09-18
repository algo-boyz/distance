#pragma once

#include "config.hpp"
#include "distance_calculator.hpp"
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

namespace vdm {

// Draw a filled semi-transparent rectangle + distance text
void drawDistanceLabel(cv::Mat& frame,
                       const cv::Rect2f& bbox,
                       float distance,
                       const cv::Scalar& color,
                       Zone zone);

// Global warning banner at the top of the frame
void drawWarningMessage(cv::Mat& frame, const std::string& message);

// Optional: draw the three ROI polygons for debugging
void drawRoiZones(cv::Mat& frame, const std::map<Zone, std::vector<cv::Point>>& zones);

} // namespace vdm
