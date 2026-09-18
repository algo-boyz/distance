#include "visualization.hpp"
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <iomanip>

namespace vdm {

void drawDistanceLabel(cv::Mat& frame,
                       const cv::Rect2f& bbox,
                       float distance,
                       const cv::Scalar& color,
                       Zone zone) {
    const int x1 = static_cast<int>(bbox.x);
    const int y1 = static_cast<int>(bbox.y);
    const int x2 = static_cast<int>(bbox.x + bbox.width);
    const int y2 = static_cast<int>(bbox.y + bbox.height);

    auto warnIt = WARNING_DISTANCES.find(zone);
    const float warnDist = (warnIt != WARNING_DISTANCES.end()) ? warnIt->second : 2.0f;

    // Semi-transparent red overlay when too close
    if (distance < warnDist) {
        cv::Mat overlay = frame.clone();
        cv::rectangle(overlay, cv::Point(x1, y1), cv::Point(x2, y2), {0, 0, 255}, -1);
        cv::addWeighted(overlay, 0.5, frame, 0.5, 0, frame);
        cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), {0, 0, 255}, 3);
    }

    // Format distance text
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << distance << "m";
    const std::string text = oss.str();

    const int fontFace   = cv::FONT_HERSHEY_DUPLEX;
    const double fontScale = 1.2;
    const int thickness  = 3;
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);

    const int frameH = frame.rows;
    const int frameW = frame.cols;
    const int centerX = (x1 + x2) / 2;
    int textX = std::max(10, std::min(centerX - textSize.width / 2, frameW - textSize.width - 20));
    int textY = std::min(frameH - 35, y2 + textSize.height + 20);

    // Background rectangle
    const int pad = 10;
    int bgX1 = std::max(0, textX - pad);
    int bgY1 = std::max(0, textY - textSize.height - pad);
    int bgX2 = std::min(frameW, textX + textSize.width + pad);
    int bgY2 = std::min(frameH, textY + pad);

    cv::Mat overlay = frame.clone();
    cv::rectangle(overlay, cv::Point(bgX1, bgY1), cv::Point(bgX2, bgY2), color, -1);
    cv::addWeighted(overlay, 0.8, frame, 0.2, 0, frame);
    cv::rectangle(frame, cv::Point(bgX1, bgY1), cv::Point(bgX2, bgY2), {255, 255, 255}, 2);

    // Text with black outline
    cv::putText(frame, text, cv::Point(textX, textY), fontFace, fontScale, {0, 0, 0}, thickness + 2);
    cv::putText(frame, text, cv::Point(textX, textY), fontFace, fontScale, {255, 255, 255}, thickness);
}

void drawWarningMessage(cv::Mat& frame, const std::string& message) {
    const int fontFace   = cv::FONT_HERSHEY_DUPLEX;
    const double fontScale = 1.0;
    const int thickness  = 3;
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(message, fontFace, fontScale, thickness, &baseline);

    const int frameW = frame.cols;
    const int frameH = frame.rows;
    int posX = frameW / 2 - textSize.width / 2;
    int posY = 80;

    const int pad = 15;
    int bgX1 = std::max(0, posX - pad);
    int bgY1 = std::max(0, posY - textSize.height - pad);
    int bgX2 = std::min(frameW, posX + textSize.width + pad);
    int bgY2 = std::min(frameH, posY + pad);

    cv::Mat overlay = frame.clone();
    cv::rectangle(overlay, cv::Point(bgX1, bgY1), cv::Point(bgX2, bgY2), {0, 0, 255}, -1);
    cv::addWeighted(overlay, 0.7, frame, 0.3, 0, frame);
    cv::rectangle(frame, cv::Point(bgX1, bgY1), cv::Point(bgX2, bgY2), {255, 255, 255}, 3);

    cv::putText(frame, message, cv::Point(posX, posY), fontFace, fontScale, {0, 0, 0}, thickness + 2);
    cv::putText(frame, message, cv::Point(posX, posY), fontFace, fontScale, {255, 255, 255}, thickness);
}

void drawRoiZones(cv::Mat& frame, const std::map<Zone, std::vector<cv::Point>>& zones) {
    const std::map<Zone, cv::Scalar> colors = {
        {Zone::LEFT,  {255, 128, 0}},
        {Zone::MAIN,  {0, 255, 255}},
        {Zone::RIGHT, {255, 0, 255}}
    };
    for (const auto& [zone, poly] : zones) {
        if (poly.empty()) continue;
        auto it = colors.find(zone);
        cv::Scalar c = (it != colors.end()) ? it->second : cv::Scalar(255, 255, 255);
        cv::polylines(frame, poly, true, c, 2);
    }
}

} // namespace vdm
