
#ifndef MEASUREMENT_GENERATOR_H
#define MEASUREMENT_GENERATOR_H

#include <string>

struct Measurement
{
    int frame_number;

    double timestamp;

    double x;
    double y;

    double width;
    double height;

    double velocity_x;
    double velocity_y;
};


bool generateMeasurements(
    const std::string& video_path,
    const std::string& csv_path,
    const std::string& annotated_video_path
);

#endif
