#include <iostream>
#include <string>

#include "fractals.hpp"

int main(int argc, char **argv)
{
    uint32_t width = 1920;  // Default width
    uint32_t height = 1080; // Default height
    uint32_t samples = 128; // Default samples

    if (argc == 2)
        samples = std::stoi(argv[1]);
    else {
        samples = 128;
    }

    Fractals fractal(samples, width, height);
    // fractal.run();
    fractal.calculateCUDA();
    
    return 0;
}
