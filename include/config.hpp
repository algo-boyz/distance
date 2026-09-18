#pragma once

#include <opencv2/core.hpp>
#include <array>
#include <map>
#include <string>
#include <vector>

namespace vdm {

// Frame resolution the original system was calibrated for
constexpr int CALIBRATION_WIDTH  = 2042;
constexpr int CALIBRATION_HEIGHT = 1148;

// Camera / geometry parameters
constexpr float FOCAL_LENGTH = 500.0f;          // pixels
constexpr float DISPLACEMENT_COEFF = 0.0001f;   // perspective correction

// Detection thresholds
constexpr float VEHICLE_CONFIDENCE      = 0.70f;
constexpr float PLATE_CONFIDENCE        = 0.475f;
constexpr float MAX_DISPLAY_DISTANCE    = 15.0f;
constexpr float MAIN_ROI_WARNING_THRESH = 5.0f;

// COCO class IDs used by the YOLO26 model
enum class VehicleClass : int {
    Car        = 2,
    Motorcycle = 3,
    Bus        = 5,
    Truck      = 7
};

inline const std::vector<int> TARGET_CLASSES = {2, 3, 5, 7};

// Real-world heights (meters)
inline const std::map<int, float> REAL_VEHICLE_HEIGHTS = {
    {2, 1.55f},  // Car
    {3, 1.20f},  // Motorcycle
    {5, 3.00f},  // Bus
    {7, 2.50f}   // Truck
};

// ROI zone names
enum class Zone { LEFT, MAIN, RIGHT, NONE };

inline const char* zoneName(Zone z) {
    switch (z) {
        case Zone::LEFT:  return "LEFT";
        case Zone::MAIN:  return "MAIN";
        case Zone::RIGHT: return "RIGHT";
        default:          return "NONE";
    }
}

// Trapezoidal ROI polygons (original 2042x1148 calibration)
// Order: top-left, top-right, bottom-right, bottom-left
inline const std::map<Zone, std::vector<cv::Point>> ROI_ZONES = {
    {Zone::LEFT,  {{240, 600}, {925, 550}, {312, 1100}, {100, 1100}}},
    {Zone::MAIN,  {{925, 550}, {1025, 550}, {1712, 1100}, {312, 1100}}},
    {Zone::RIGHT, {{1025, 550}, {1802, 600}, {1942, 1100}, {1712, 1100}}}
};

// Optical centers used for perspective correction (from original Python)
inline const std::map<Zone, cv::Point2f> OPTICAL_CENTERS = {
    {Zone::LEFT,  {156.f, 1050.f}},
    {Zone::MAIN,  {1000.f, 1020.f}},
    {Zone::RIGHT, {1868.f, 1050.f}}
};

// Warning distances (meters)
inline const std::map<Zone, float> WARNING_DISTANCES = {
    {Zone::LEFT,  1.0f},
    {Zone::MAIN,  2.0f},
    {Zone::RIGHT, 1.0f}
};

// Maximum distance to display a label for side lanes
inline const std::map<Zone, float> DISTANCE_DISPLAY_THRESHOLDS = {
    {Zone::LEFT,  5.0f},
    {Zone::RIGHT, 5.0f}
    // MAIN uses MAX_DISPLAY_DISTANCE
};

} // namespace vdm
