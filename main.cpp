#include <iostream>
#include <string>
// #include <cstring>

#include "fractals.hpp"


void printHelp() {
    std::cout << "Usage: Fractals.exe [options]\n"
              << "Options:\n"
              << "  -i <samples>    Number of samples (default: 128)\n"
              << "  -d <device>     Device to use (CPU or CUDA, default: CUDA)\n"
              << "  -w <width>      Width of the image (default: 1920)\n"
              << "  -h <height>     Height of the image (default: 1080)\n"
              << "  -h              Show this help message\n";
}

int main(int argc, char **argv)
{
    uint32_t width = 1920;  // Default width
    uint32_t height = 1080; // Default height
    uint32_t samples = 128; // Default samples
    std::string device = "CPU"; // Default device

    for (int i = 1; i < argc; ++i) 
    {
        if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc)
            samples = std::stoi(argv[++i]);
        
        else if (std::strcmp(argv[i], "-d") == 0 && i + 1 < argc)
            device = argv[++i];
        
        else if (std::strcmp(argv[i], "-w") == 0 && i + 1 < argc)
            width = std::stoi(argv[++i]);
        
        else if (std::strcmp(argv[i], "-h") == 0 && i + 1 < argc)
            height = std::stoi(argv[++i]);
        
        else if (std::strcmp(argv[i], "--help") == 0)
        {
            printHelp();
            return 0;
        }
    }

    Fractals fractal(samples, width, height);

    if (device == "CPU") {
        fractal.run();
    } else {
        fractal.calculateCUDA();
    }
    
    return 0;
}
