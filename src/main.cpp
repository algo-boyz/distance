/**
 * Vehicle Distance Measurement System - C++ port
 *
 * Original Python: https://github.com/kemalkilicaslan/Vehicle-Distance-Measurement-System
 *
 * This port keeps the same geometry, ROI zones, distance formula and
 * visualisation style. Detection uses OpenCV DNN with ONNX models
 * exported from Ultralytics YOLO26.
 *
 * Build (Apple Silicon / M4):
 *   mkdir build && cd build
 *   cmake -DCMAKE_BUILD_TYPE=Release ..
 *   make -j$(sysctl -n hw.ncpu)
 *
 * Export models first (Python side):
 *   yolo export model=yolo26x.pt format=onnx imgsz=640
 *   yolo export model=vehicle-plate.pt format=onnx imgsz=640
 *
 * Usage:
 *   ./vehicle_distance --video dashcam_video.mov \
 *                      --vehicle-model models/yolo26x.onnx \
 *                      --plate-model models/vehicle-plate.onnx \
 *                      --output Vehicle-Distance-Measurement.mp4
 */

#include "config.hpp"
#include "roi_utils.hpp"
#include "distance_calculator.hpp"
#include "visualization.hpp"
#include "yolo_detector.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <memory>

using namespace vdm;

struct Args {
    std::string videoPath    = "dashcam_video.mov";
    std::string vehicleModel = "models/yolo26x.onnx";
    std::string plateModel;   // empty = disabled (optional)
    std::string outputPath   = "Vehicle-Distance-Measurement.mp4";
    bool showWindow = true;
    bool drawRoi    = false;
    bool enablePlates = false; // set true only when a plate model path is given
};

Args parseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << std::endl;
                std::exit(1);
            }
            return argv[++i];
        };
        if (arg == "--video" || arg == "-v")          a.videoPath    = next(arg.c_str());
        else if (arg == "--vehicle-model")            a.vehicleModel = next(arg.c_str());
        else if (arg == "--plate-model") {
            a.plateModel = next(arg.c_str());
            a.enablePlates = !a.plateModel.empty() && a.plateModel != "none";
        }
        else if (arg == "--output" || arg == "-o")    a.outputPath   = next(arg.c_str());
        else if (arg == "--no-window")                a.showWindow   = false;
        else if (arg == "--draw-roi")                 a.drawRoi      = true;
        else if (arg == "--no-plates")                a.enablePlates = false;
        else if (arg == "--help" || arg == "-h") {
            std::cout <<
                "Usage: vehicle_distance [options]\n"
                "  --video PATH          Input video (default: dashcam_video.mov)\n"
                "  --vehicle-model PATH  YOLO26 ONNX (default: models/yolo26x.onnx)\n"
                "  --plate-model PATH    Plate detector ONNX (optional; omit or use 'none' to skip)\n"
                "  --no-plates           Disable plate blurring even if a model is given\n"
                "  --output PATH         Output video (default: Vehicle-Distance-Measurement.mp4)\n"
                "  --no-window           Do not open preview window\n"
                "  --draw-roi            Draw ROI polygons for debugging\n";
            std::exit(0);
        }
    }
    return a;
}

// Blur license plates inside a vehicle bounding box using a second detector.
// Returns false if the detector threw (e.g. incompatible ONNX graph) so the
// caller can disable plate processing for the rest of the run.
bool blurPlatesInVehicle(cv::Mat& frame,
                         const cv::Rect2f& vehicleBbox,
                         YoloDetector& plateDetector) {
    int x1 = std::max(0, static_cast<int>(vehicleBbox.x));
    int y1 = std::max(0, static_cast<int>(vehicleBbox.y));
    int x2 = std::min(frame.cols, static_cast<int>(vehicleBbox.x + vehicleBbox.width));
    int y2 = std::min(frame.rows, static_cast<int>(vehicleBbox.y + vehicleBbox.height));

    if (x2 <= x1 || y2 <= y1) return true;

    cv::Mat roi = frame(cv::Rect(x1, y1, x2 - x1, y2 - y1));
    if (roi.empty()) return true;

    std::vector<Box> plates;
    try {
        // Detect plates inside the vehicle crop (no class filter – specialised model)
        plates = plateDetector.detect(roi, {});
    } catch (const cv::Exception& e) {
        std::cerr << "[plate] OpenCV exception during detect – disabling plate blur: "
                  << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "[plate] exception during detect – disabling plate blur: "
                  << e.what() << std::endl;
        return false;
    }

    for (const auto& p : plates) {
        if (p.conf < PLATE_CONFIDENCE) continue;

        int px1 = std::max(0, static_cast<int>(p.x1) + x1);
        int py1 = std::max(0, static_cast<int>(p.y1) + y1);
        int px2 = std::min(frame.cols, static_cast<int>(p.x2) + x1);
        int py2 = std::min(frame.rows, static_cast<int>(p.y2) + y1);

        if (px2 <= px1 || py2 <= py1) continue;

        cv::Mat plateRoi = frame(cv::Rect(px1, py1, px2 - px1, py2 - py1));
        if (plateRoi.empty()) continue;

        cv::Mat blurred;
        cv::GaussianBlur(plateRoi, blurred, cv::Size(51, 51), 30);
        blurred.copyTo(plateRoi);
    }
    return true;
}

int main(int argc, char** argv) {
    Args args = parseArgs(argc, argv);

    // ---- Load detectors ----
    YoloDetector vehicleDetector(args.vehicleModel, VEHICLE_CONFIDENCE);
    if (!vehicleDetector.isLoaded()) {
        std::cerr << "Failed to load vehicle model. Export with:\n"
                  << "  yolo export model=yolo26x.pt format=onnx imgsz=640\n";
        return 1;
    }

    // Plate detector is optional. RF-DETR / non-YOLO ONNX graphs are not supported
    // by the current post-processor – omit --plate-model (or pass "none") to skip.
    std::unique_ptr<YoloDetector> plateDetector;
    bool platesEnabled = args.enablePlates;
    if (platesEnabled) {
        plateDetector = std::make_unique<YoloDetector>(args.plateModel, PLATE_CONFIDENCE);
        if (!plateDetector->isLoaded()) {
            std::cerr << "Warning: plate model failed to load. Plate blurring disabled.\n";
            platesEnabled = false;
            plateDetector.reset();
        } else {
            std::cout << "Plate blurring enabled: " << args.plateModel << std::endl;
        }
    } else {
        std::cout << "Plate blurring disabled (no --plate-model given).\n";
    }

    // ---- Open video ----
    cv::VideoCapture cap(args.videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video: " << args.videoPath << std::endl;
        return 1;
    }

    const int frameW = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int frameH = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    const double fps = cap.get(cv::CAP_PROP_FPS);
    const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');

    cv::VideoWriter writer(args.outputPath, fourcc, fps > 0 ? fps : 30.0,
                           cv::Size(frameW, frameH));
    if (!writer.isOpened()) {
        std::cerr << "Cannot open output video writer: " << args.outputPath << std::endl;
        return 1;
    }

    // Scale ROI polygons to the actual frame size
    auto scaledRois = scaleRoiZones(frameW, frameH);

    std::cout << "Processing " << args.videoPath
              << " (" << frameW << "x" << frameH << " @ " << fps << " fps)\n"
              << "Output -> " << args.outputPath << std::endl;

    cv::Mat frame;
    int frameIdx = 0;
    auto t0 = std::chrono::steady_clock::now();

    while (cap.read(frame)) {
        if (frame.empty()) break;
        ++frameIdx;

        cv::Mat annotated = frame.clone();

        // 1. Detect vehicles
        auto boxes = vehicleDetector.detect(annotated, TARGET_CLASSES);

        bool anyTooClose = false;

        for (const auto& b : boxes) {
            cv::Rect2f bbox(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
            cv::Point2f center((b.x1 + b.x2) * 0.5f,
                               (b.y1 + b.y2) * 0.5f);

            // 2. Blur plates (privacy) — skipped when disabled or after a runtime failure
            if (platesEnabled && plateDetector) {
                if (!blurPlatesInVehicle(annotated, bbox, *plateDetector)) {
                    platesEnabled = false; // disable for remaining frames
                    std::cerr << "Plate blurring disabled for the rest of this run.\n";
                }
            }

            // 3. ROI classification (use original-resolution logic on scaled ROIs)
            // For simplicity we re-implement the test against scaled polygons
            Zone zone = Zone::NONE;
            for (const auto& [z, poly] : scaledRois) {
                if (isPointInRoi(center, poly)) {
                    zone = z;
                    break;
                }
            }
            if (zone == Zone::NONE || b.conf < VEHICLE_CONFIDENCE) continue;

            // 4. Distance
            // Note: optical centres are still in the original calibration space.
            // For best results the input video should be close to 2042x1148.
            // Otherwise you may want to scale the optical centres as well.
            float distance = calculateDistance(bbox, b.classId, zone);

            auto warnIt = WARNING_DISTANCES.find(zone);
            float warnDist = (warnIt != WARNING_DISTANCES.end()) ? warnIt->second : 2.0f;
            if (distance < warnDist) anyTooClose = true;

            // 5. Draw label when appropriate
            if (shouldDisplayDistance(distance, zone) && distance <= MAX_DISPLAY_DISTANCE) {
                cv::Scalar color = getDistanceColor(distance, zone);
                drawDistanceLabel(annotated, bbox, distance, color, zone);
            }
        }

        // Global warning banner
        if (anyTooClose) {
            drawWarningMessage(annotated, "WARNING: VEHICLE TOO CLOSE!");
        }

        if (args.drawRoi) {
            drawRoiZones(annotated, scaledRois);
        }

        writer.write(annotated);

        if (args.showWindow) {
            cv::imshow("Vehicle Distance Measurement", annotated);
            if (cv::waitKey(1) == 'q') break;
        }

        if (frameIdx % 30 == 0) {
            auto t1 = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(t1 - t0).count();
            std::cout << "Frame " << frameIdx
                      << "  avg FPS: " << (frameIdx / elapsed) << std::endl;
        }
    }

    cap.release();
    writer.release();
    cv::destroyAllWindows();

    auto tEnd = std::chrono::steady_clock::now();
    double total = std::chrono::duration<double>(tEnd - t0).count();
    std::cout << "Done. Processed " << frameIdx << " frames in "
              << total << " s (" << (frameIdx / total) << " FPS avg)\n"
              << "Saved: " << args.outputPath << std::endl;
    return 0;
}
