#include "measurement_generator.h"

#include <opencv2/opencv.hpp>

#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>

bool generateMeasurements(
    const std::string& video_path,
    const std::string& csv_path,
    const std::string& annotated_video_path
)
{
    cv::VideoCapture video(video_path);

    if (!video.isOpened())
    {
        std::cerr << "ERROR: Cannot open input video.\n";
        return false;
    }

    double fps = video.get(cv::CAP_PROP_FPS);

    if (fps <= 0.0)
    {
        fps = 30.0;
    }

    int frame_width =
        static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH));

    int frame_height =
        static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT));

    std::ofstream csv(csv_path);

    if (!csv.is_open())
    {
        std::cerr << "ERROR: Cannot create CSV file.\n";
        return false;
    }

    csv << "frame,timestamp,x,y,width,height,"
           "velocity_x,velocity_y\n";

    cv::Ptr<cv::BackgroundSubtractor> background_subtractor =
        cv::createBackgroundSubtractorMOG2(
            500,       
            16,       
            false      
        );

    cv::VideoWriter output_video;

    if (!annotated_video_path.empty())
    {
        output_video.open(
            annotated_video_path,
            cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
            fps,
            cv::Size(frame_width, frame_height)
        );

        if (!output_video.isOpened())
        {
            std::cerr << "ERROR: Cannot create output video.\n";
            return false;
        }
    }

    cv::Mat frame;
    cv::Mat foreground_mask;

    int frame_number = 0;

    bool previous_detection = false;

    double previous_x = 0.0;
    double previous_y = 0.0;

    std::cout << "Starting video processing...\n";

    while (video.read(frame))
    {
        frame_number++;

        double timestamp =
            static_cast<double>(frame_number - 1) / fps;

        background_subtractor->apply(
            frame,
            foreground_mask
        );

        cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(5, 5)
        );

        cv::morphologyEx(
            foreground_mask,
            foreground_mask,
            cv::MORPH_OPEN,
            kernel
        );

        cv::morphologyEx(
            foreground_mask,
            foreground_mask,
            cv::MORPH_CLOSE,
            kernel
        );

        std::vector<std::vector<cv::Point>> contours;

        cv::findContours(
            foreground_mask,
            contours,
            cv::RETR_EXTERNAL,
            cv::CHAIN_APPROX_SIMPLE
        );

        double largest_area = 0.0;
        int best_contour_index = -1;

        for (int i = 0; i < static_cast<int>(contours.size()); i++)
        {
            double area = cv::contourArea(contours[i]);

            if (area > largest_area && area >= 500.0)
            {
                largest_area = area;
                best_contour_index = i;
            }
        }

        bool current_detection = false;

        if (best_contour_index >= 0)
        {
            current_detection = true;

            cv::Rect bounding_box =
                cv::boundingRect(contours[best_contour_index]);

            cv::Moments moments =
                cv::moments(contours[best_contour_index]);

            if (moments.m00 != 0.0)
            {
                double x = moments.m10 / moments.m00;
                double y = moments.m01 / moments.m00;

                double velocity_x = 0.0;
                double velocity_y = 0.0;

                if (previous_detection)
                {
                    double dt = 1.0 / fps;

                    velocity_x = (x - previous_x) / dt;
                    velocity_y = (y - previous_y) / dt;
                }

                csv << frame_number << ","
                    << timestamp << ","
                    << x << ","
                    << y << ","
                    << bounding_box.width << ","
                    << bounding_box.height << ","
                    << velocity_x << ","
                    << velocity_y << "\n";

                cv::rectangle(
                    frame,
                    bounding_box,
                    cv::Scalar(0, 255, 0),
                    2
                );

                cv::circle(
                    frame,
                    cv::Point(
                        static_cast<int>(x),
                        static_cast<int>(y)
                    ),
                    5,
                    cv::Scalar(0, 0, 255),
                    -1
                );

                cv::putText(
                    frame,
                    "Moving Object",
                    cv::Point(bounding_box.x, bounding_box.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.6,
                    cv::Scalar(0, 255, 0),
                    2
                );

                previous_x = x;
                previous_y = y;
            }
            else
            {
                current_detection = false;
            }
        }

        if (!current_detection)
        {
            previous_detection = false;
        }
        else
        {
            previous_detection = true;
        }

        if (output_video.isOpened())
        {
            output_video.write(frame);
        }

        if (frame_number % 100 == 0)
        {
            std::cout
                << "Processed frames: "
                << frame_number << "\n";
        }
    }

    video.release();
    output_video.release();
    csv.close();

    std::cout << "\nProcessing completed.\n";
    std::cout << "Total frames: " << frame_number << "\n";
    std::cout << "CSV saved: " << csv_path << "\n";

    return true;
}
