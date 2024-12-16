#include "fractals.hpp"

#include <SFML/Graphics.hpp>

#include <GL/gl.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuComplex.h>

#include <stdio.h>

__global__ void calculateFractalKernel(unsigned char *pixelData, uint32_t WIDTH, uint32_t HEIGHT, sf::Color *paletteData, size_t paletteSize,
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

        sf::Color pixelColor = (n == MAX_ITERATIONS) ? sf::Color::Black : paletteData[n % paletteSize];
        pixelData[4 * (y * WIDTH + x) + 0] = pixelColor.r;
        pixelData[4 * (y * WIDTH + x) + 1] = pixelColor.g;
        pixelData[4 * (y * WIDTH + x) + 2] = pixelColor.b;
    }
}


void Fractals::calculateCUDA()
{
    // Create sf window and display a red image
    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Fractals");
    window.setFramerateLimit(60);

    // Create an image
    const uint32_t imageSize = WIDTH * HEIGHT;

    // Create a texture and sprite
    sf::Texture texture;
    texture.create(WIDTH, HEIGHT);
    
    
    // Get texture ID
    GLuint textureID = texture.getNativeHandle();
    
    // Register texture with CUDA
    cudaGraphicsResource *cudaResource;
    cudaGraphicsRegisterImage(&cudaResource, textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
    
    
    // Set up GRID & BLOCK Dimensions
    dim3 blockDIM(32, 32);
    dim3 gridDIM( (WIDTH + blockDIM.x - 1) / blockDIM.x, 
                  (HEIGHT + blockDIM.y - 1) / blockDIM.y );
    
    // Setup and Allocate Memory for Color Palette
    sf::Color *paletteData;
    cudaMalloc(&paletteData, sizeof(sf::Color) * palette.colors.size());
    cudaMemcpy(paletteData, palette.colors.data(), sizeof(sf::Color) * palette.colors.size(), cudaMemcpyHostToDevice);
    
    // Window Loop
    sf::Sprite sprite(texture);

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        cudaGraphicsMapResources(1, &cudaResource);
        cudaArray* cudaArray;
        cudaGraphicsSubResourceGetMappedArray(&cudaArray, cudaResource, 0, 0);


        unsigned char* pixelData;
        cudaMemcpyFromArray(&pixelData, cudaArray, 0, 0, imageSize, cudaMemcpyDeviceToHost);

        // Launch Kernel
        calculateFractalKernel<<<gridDIM, blockDIM>>>(pixelData, WIDTH, HEIGHT, paletteData, palette.colors.size(), min_Re, max_Re, min_Im, max_Im, MAX_ITERATIONS);

        cudaGraphicsUnmapResources(1, &cudaResource);

        window.clear();
        window.draw(sprite);
        window.display();
    }

    // Unregister texture with CUDA
    cudaFree(paletteData);
    cudaGraphicsUnregisterResource(cudaResource);
}