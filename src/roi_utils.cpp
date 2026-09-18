#include "roi_utils.hpp"
#include <cmath>

namespace vdm {

// Ray-casting point-in-polygon (avoids OpenCV pointPolygonTest, which some
// Homebrew / modular OpenCV builds do not expose cleanly).
bool isPointInRoi(const cv::Point2f& point, const std::vector<cv::Point>& polygon) {
    const size_t n = polygon.size();
    if (n < 3) return false;

    bool inside = false;
    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        const float xi = static_cast<float>(polygon[i].x);
        const float yi = static_cast<float>(polygon[i].y);
        const float xj = static_cast<float>(polygon[j].x);
        const float yj = static_cast<float>(polygon[j].y);

        if (yi == yj) continue; // horizontal edge — skip to avoid div-by-zero
        const bool intersect =
            ((yi > point.y) != (yj > point.y)) &&
            (point.x < (xj - xi) * (point.y - yi) / (yj - yi) + xi);

        if (intersect) inside = !inside;
    }
    return inside;
}

Zone getVehicleRoiZone(const cv::Point2f& center) {
    for (const auto& [zone, poly] : ROI_ZONES) {
        if (isPointInRoi(center, poly)) {
            return zone;
        }
    }
    return Zone::NONE;
}

std::map<Zone, std::vector<cv::Point>> scaleRoiZones(int frameWidth, int frameHeight) {
    const float sx = static_cast<float>(frameWidth)  / CALIBRATION_WIDTH;
    const float sy = static_cast<float>(frameHeight) / CALIBRATION_HEIGHT;

    std::map<Zone, std::vector<cv::Point>> scaled;
    for (const auto& [zone, poly] : ROI_ZONES) {
        std::vector<cv::Point> pts;
        pts.reserve(poly.size());
        for (const auto& p : poly) {
            pts.emplace_back(static_cast<int>(std::round(p.x * sx)),
                             static_cast<int>(std::round(p.y * sy)));
        }
        scaled[zone] = std::move(pts);
    }
    return scaled;
}

} // namespace vdm
