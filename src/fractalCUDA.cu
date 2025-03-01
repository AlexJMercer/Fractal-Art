#include "../lib/fractals.hpp"

#include <SFML/Graphics.hpp>

#include <cuda_runtime.h>

#include <cuComplex.h>

#include <stdio.h>

__global__ void calculateFractalKernel(unsigned char *pixelData, uint32_t WIDTH, uint32_t HEIGHT, sf::Color *paletteData, sf::Color blackColor, size_t paletteSize,
                                       double min_Re, double max_Re, double min_Im, double max_Im, uint32_t MAX_ITERATIONS)
{
  uint16_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint16_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < WIDTH && y < HEIGHT)
  {
    double real = min_Re + x * (max_Re - min_Re) / WIDTH;
    double imag = min_Im + y * (max_Im - min_Im) / HEIGHT;

    cuDoubleComplex constant = make_cuDoubleComplex(real, imag);
    cuDoubleComplex z = make_cuDoubleComplex(0, 0);

    uint16_t n = 0;
    for (; n < MAX_ITERATIONS; ++n)
    {
      if (cuCabs(z) > 2.0)
        break;
      z = cuCadd(cuCmul(z, z), constant);
    }

    sf::Color pixelColor = (n == MAX_ITERATIONS) ? blackColor : paletteData[n % paletteSize];
    pixelData[4 * (y * WIDTH + x) + 0] = pixelColor.r;
    pixelData[4 * (y * WIDTH + x) + 1] = pixelColor.g;
    pixelData[4 * (y * WIDTH + x) + 2] = pixelColor.b;
    pixelData[4 * (y * WIDTH + x) + 3] = pixelColor.a;
  }
}

void Fractals::calculateCUDA()
{
  // Create sf window and display a red image
  sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Fractals");
  window.setFramerateLimit(30);

  // Create an image
  const uint32_t imageSize = WIDTH * HEIGHT * 4;
  unsigned char *pixelData;
  cudaMallocManaged(&pixelData, imageSize);

  // Set up GRID & BLOCK Dimensions
  dim3 blockDIM(32, 32);
  dim3 gridDIM((WIDTH + blockDIM.x - 1) / blockDIM.x,
               (HEIGHT + blockDIM.y - 1) / blockDIM.y);

  // Setup and Allocate Memory for Color Palette
  sf::Color *paletteData;
  cudaMallocManaged(&paletteData, sizeof(sf::Color) * palette.colors.size());
  cudaMemcpy(paletteData, palette.colors.data(), sizeof(sf::Color) * palette.colors.size(), cudaMemcpyHostToDevice);

  // Define Black Color
  sf::Color blackColor = sf::Color(0, 0, 0, 255);

  // Window Loop
  sf::Texture texture;
  texture.create(WIDTH, HEIGHT);
  sf::Sprite sprite(texture);

  // First display the initial fractal
  calculateFractalKernel<<<gridDIM, blockDIM>>>(pixelData, WIDTH, HEIGHT, paletteData, blackColor, palette.colors.size(), min_Re, max_Re, min_Im, max_Im, MAX_ITERATIONS);
  cudaDeviceSynchronize();

  texture.update(pixelData);

  // One bool to rule them all
  bool needsRecalculation = false;

  while (window.isOpen())
  {
    sf::Event event;
    while (window.pollEvent(event))
    {
      if (event.type == sf::Event::Closed)
        window.close();

      if (event.type == sf::Event::KeyPressed)
      {
        // Delta for Panning
        double X = (max_Re - min_Re) * deltaX * 0.1;
        double Y = (max_Im - min_Im) * deltaY * 0.1;

        switch (event.key.code)
        {
        case sf::Keyboard::Escape:
          window.close();
          break;

        case sf::Keyboard::Up:
          max_Im -= Y;
          min_Im -= Y;
          needsRecalculation = true;
          break;

        case sf::Keyboard::Down:
          max_Im += Y;
          min_Im += Y;
          needsRecalculation = true;
          break;

        case sf::Keyboard::Left:
          max_Re -= X;
          min_Re -= X;
          needsRecalculation = true;
          break;

        case sf::Keyboard::Right:
          max_Re += X;
          min_Re += X;
          needsRecalculation = true;
          break;

          // case sf::Keyboard::Space:
          //     image.saveToFile("fractalGPU.jpg");
          //     break;

        default:
          break;
        }
      }

      if (event.type == sf::Event::MouseWheelScrolled)
      {
        if (event.mouseWheelScroll.delta > 0)
          MAX_ITERATIONS *= 1.2;
        else
          MAX_ITERATIONS /= 1.2;

        if (MAX_ITERATIONS < 1)
          MAX_ITERATIONS = 1;

        needsRecalculation = true;
      }

      // Mark center and Zoom In/Out
      if (event.type == sf::Event::MouseButtonPressed)
      {
        auto zoom = [this](double z, double mouseX, double mouseY)
        {
          // Calculate the new center based on the mouse click position
          std::complex<double> constant(min_Re + (max_Re - min_Re) * mouseX / WIDTH,
                                        min_Im + (max_Im - min_Im) * mouseY / HEIGHT);

          double newMinRe = constant.real() - (max_Re - min_Re) / 2 / z;
          max_Re = constant.real() + (max_Re - min_Re) / 2 / z;
          min_Re = newMinRe;

          double newMinIm = constant.imag() - (max_Im - min_Im) / 2 / z;
          max_Im = constant.imag() + (max_Im - min_Im) / 2 / z;
          min_Im = newMinIm;
        };

        // Zoom In where clicked
        if (event.mouseButton.button == sf::Mouse::Left)
        {
          zoom(2, event.mouseButton.x, event.mouseButton.y);
          needsRecalculation = true;
        }

        // Zoom out
        if (event.mouseButton.button == sf::Mouse::Right)
        {
          zoom(0.5, event.mouseButton.x, event.mouseButton.y);
          needsRecalculation = true;
        }
      }
    }

    if (needsRecalculation)
    {
      // Launch Kernel
      calculateFractalKernel<<<gridDIM, blockDIM>>>(pixelData, WIDTH, HEIGHT, paletteData, blackColor, palette.colors.size(), min_Re, max_Re, min_Im, max_Im, MAX_ITERATIONS);
      cudaDeviceSynchronize();

      // Update texture with pixel data
      texture.update(pixelData);

      needsRecalculation = false;
    }

    window.clear();
    window.draw(sprite);
    window.display();
  }

  // Free allocated memory
  cudaFree(paletteData);
  cudaFree(pixelData);
}
