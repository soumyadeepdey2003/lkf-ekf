#include "measurement_generator.h"

#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
    if (argc != 4)
    {
        std::cout << "Usage:\n";
        std::cout << "measurement_generator_cpu "
                  << "<input_video> <output_csv> <output_video>\n";

        return 1;
    }

    std::string video_path = argv[1];
    std::string csv_path = argv[2];
    std::string annotated_video_path = argv[3];

    bool success = generateMeasurements(
        video_path,
        csv_path,
        annotated_video_path
    );

    if (!success)
    {
        std::cerr << "Measurement generation failed.\n";
        return 1;
    }

    std::cout << "Measurement generation successful.\n";

    return 0;
}
