#pragma once

#include <SFML/Graphics.hpp>

#include <device_launch_parameters.h>

#include "color.hpp"

#include <thread>
#include <string>
#include <vector>
#include <complex>
#include <cmath>
#include <random>
#include <chrono>

class Fractals
{
public:
  Fractals(uint32_t SAMPLES, uint32_t WIDTH, uint32_t HEIGHT);
  ~Fractals();

  void run();
  void calculateFractal(sf::Image &);

  // For CUDA
  void calculateCUDA();

private:
  ColorPalette palette;

  uint32_t SAMPLES;
  uint32_t WIDTH;
  uint32_t HEIGHT;

  uint32_t MAX_ITERATIONS;

  // Initialize Multi-threading
  const uint32_t numThreads = std::thread::hardware_concurrency();
  std::vector<std::thread> threads;
  const uint32_t rowsPerThread = HEIGHT / numThreads;

  // Variables for the Fractal rendering window
  double min_Re;
  double max_Re;
  double min_Im;
  double max_Im;

  // Properties for Pan, Zoom, Scroll
  double zoomFactor;
  double deltaX;
  double deltaY;
};
