#include <iostream>

#include "fractals.cpp"

int main(int argc, char *argv[])
{
    uint32_t WIDTH = 1920;  // Default width
    uint32_t HEIGHT = 1080; // Default height
    uint32_t SAMPLES = 128; // Default samples

    if (argc == 2)
        SAMPLES = std::stoi(argv[1]);
    else {
        SAMPLES = 128;
    }

    Fractals fractal(SAMPLES, WIDTH, HEIGHT);
    fractal.run();
    
    return 0;
}
