#pragma once

#include "config.hpp"
#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace vdm {

struct Box {
    float x1, y1, x2, y2;
    float conf;
    int   classId;
};

class YoloDetector {
public:
    // Load an ONNX model exported from Ultralytics
    // Example: yolo export model=yolo12x.pt format=onnx imgsz=640
    explicit YoloDetector(const std::string& onnxPath,
                          float confThreshold = VEHICLE_CONFIDENCE,
                          float nmsThreshold  = 0.45f,
                          int   inputSize     = 640);

    // Run detection on a BGR frame. Returns boxes in original image coordinates.
    std::vector<Box> detect(const cv::Mat& frame,
                            const std::vector<int>& classFilter = TARGET_CLASSES);

    bool isLoaded() const { return !net.empty(); }

private:
    cv::dnn::Net net;
    float confThreshold_;
    float nmsThreshold_;
    int   inputSize_;
    std::vector<std::string> outputNames_;

    cv::Mat preprocess(const cv::Mat& frame, float& scale, int& padX, int& padY);
    std::vector<Box> postprocess(const cv::Mat& output,
                                 float scale, int padX, int padY,
                                 int origW, int origH,
                                 const std::vector<int>& classFilter);
};

} // namespace vdm
