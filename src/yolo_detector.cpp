#include "yolo_detector.hpp"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <algorithm>

namespace vdm {

YoloDetector::YoloDetector(const std::string& onnxPath,
                           float confThreshold,
                           float nmsThreshold,
                           int inputSize)
    : confThreshold_(confThreshold)
    , nmsThreshold_(nmsThreshold)
    , inputSize_(inputSize)
{
    try {
        net = cv::dnn::readNetFromONNX(onnxPath);
        // Prefer Apple Silicon acceleration when available
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU); // change to DNN_TARGET_OPENCL if OpenCL is enabled

        // For macOS with Apple Silicon you can also try:
        // net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        // net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        outputNames_ = net.getUnconnectedOutLayersNames();
        std::cout << "[YoloDetector] Loaded model: " << onnxPath << std::endl;
    } catch (const cv::Exception& e) {
        std::cerr << "[YoloDetector] Failed to load ONNX model: " << e.what() << std::endl;
    }
}

cv::Mat YoloDetector::preprocess(const cv::Mat& frame, float& scale, int& padX, int& padY) {
    // Letterbox resize to keep aspect ratio
    const int w = frame.cols;
    const int h = frame.rows;
    scale = std::min(static_cast<float>(inputSize_) / w,
                     static_cast<float>(inputSize_) / h);
    const int newW = static_cast<int>(std::round(w * scale));
    const int newH = static_cast<int>(std::round(h * scale));
    padX = (inputSize_ - newW) / 2;
    padY = (inputSize_ - newH) / 2;

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(newW, newH));

    cv::Mat padded(inputSize_, inputSize_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(padX, padY, newW, newH)));

    cv::Mat blob;
    cv::dnn::blobFromImage(padded, blob, 1.0 / 255.0, cv::Size(inputSize_, inputSize_),
                           cv::Scalar(), true, false, CV_32F);
    return blob;
}

std::vector<Box> YoloDetector::postprocess(const cv::Mat& output,
                                           float scale, int padX, int padY,
                                           int origW, int origH,
                                           const std::vector<int>& classFilter) {
    // Ultralytics ONNX output is typically [1, 84, 8400] or [1, 8400, 84]
    // We handle both layouts.
    cv::Mat out = output;
    if (out.dims == 3) {
        // squeeze batch dimension if present
        if (out.size[0] == 1) {
            out = out.reshape(1, {out.size[1], out.size[2]});
        }
    }

    // Now out is 2-D: either (num_classes+4, num_preds) or (num_preds, num_classes+4)
    const bool transposed = (out.rows < out.cols); // common case: 84 x 8400
    const int numPreds   = transposed ? out.cols : out.rows;
    const int numAttrs   = transposed ? out.rows : out.cols; // 4 + num_classes

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> classIds;

    for (int i = 0; i < numPreds; ++i) {
        float cx, cy, w, h;
        const float* row;
        if (transposed) {
            // columns are predictions
            cx = out.at<float>(0, i);
            cy = out.at<float>(1, i);
            w  = out.at<float>(2, i);
            h  = out.at<float>(3, i);
        } else {
            cx = out.at<float>(i, 0);
            cy = out.at<float>(i, 1);
            w  = out.at<float>(i, 2);
            h  = out.at<float>(i, 3);
        }

        // Find best class score
        float maxScore = 0.f;
        int bestClass  = -1;
        for (int c = 4; c < numAttrs; ++c) {
            float s = transposed ? out.at<float>(c, i) : out.at<float>(i, c);
            if (s > maxScore) {
                maxScore = s;
                bestClass = c - 4;
            }
        }

        if (maxScore < confThreshold_) continue;

        // Optional class filter
        if (!classFilter.empty()) {
            bool allowed = false;
            for (int allowedId : classFilter) {
                if (bestClass == allowedId) { allowed = true; break; }
            }
            if (!allowed) continue;
        }

        // Undo letterbox
        float x1 = (cx - w * 0.5f - padX) / scale;
        float y1 = (cy - h * 0.5f - padY) / scale;
        float x2 = (cx + w * 0.5f - padX) / scale;
        float y2 = (cy + h * 0.5f - padY) / scale;

        // Clip to image
        x1 = std::max(0.f, std::min(x1, static_cast<float>(origW - 1)));
        y1 = std::max(0.f, std::min(y1, static_cast<float>(origH - 1)));
        x2 = std::max(0.f, std::min(x2, static_cast<float>(origW - 1)));
        y2 = std::max(0.f, std::min(y2, static_cast<float>(origH - 1)));

        if (x2 <= x1 || y2 <= y1) continue;

        boxes.emplace_back(static_cast<int>(x1), static_cast<int>(y1),
                           static_cast<int>(x2 - x1), static_cast<int>(y2 - y1));
        scores.push_back(maxScore);
        classIds.push_back(bestClass);
    }

    // NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, confThreshold_, nmsThreshold_, indices);

    std::vector<Box> result;
    result.reserve(indices.size());
    for (int idx : indices) {
        const auto& r = boxes[idx];
        result.push_back({
            static_cast<float>(r.x),
            static_cast<float>(r.y),
            static_cast<float>(r.x + r.width),
            static_cast<float>(r.y + r.height),
            scores[idx],
            classIds[idx]
        });
    }
    return result;
}

std::vector<Box> YoloDetector::detect(const cv::Mat& frame,
                                      const std::vector<int>& classFilter) {
    if (net.empty() || frame.empty()) return {};

    float scale = 1.f;
    int padX = 0, padY = 0;
    cv::Mat blob = preprocess(frame, scale, padX, padY);

    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, outputNames_);

    if (outputs.empty()) return {};

    // Take the first (and usually only) output
    return postprocess(outputs[0], scale, padX, padY,
                       frame.cols, frame.rows, classFilter);
}

} // namespace vdm
