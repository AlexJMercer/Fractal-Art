#include "fractals.hpp"


Fractals::Fractals(uint32_t SAMPLES, uint32_t WIDTH, uint32_t HEIGHT)
    : HEIGHT(HEIGHT), 
      WIDTH(WIDTH), 
      MAX_ITERATIONS(SAMPLES),
      min_Re(-2.5), 
      max_Re(1.0), 
      min_Im(-1.0), 
      max_Im(1.0),
      zoomFactor(1.0),
      deltaX(0.5),
      deltaY(0.5),
      palette(ColorPalette())
{
    // Generate a gradient palette
    palette.generateGradientPalette();
}

Fractals::~Fractals()
{
    for (auto &t : threads)
        t.detach();
        
    threads.clear();
}


void Fractals::calculateFractal(sf::Image &image)
{
    auto worker = [&](uint32_t startY, uint32_t endY) {
        for (uint32_t y = startY; y < endY; y++)
        {
            for (uint32_t x = 0; x < WIDTH; x++)
            {
                std::complex<double> constant(min_Re + (max_Re - min_Re) * x / WIDTH,
                                              min_Im + (max_Im - min_Im) * y / HEIGHT);

                std::complex<double> z{0.0, 0.0};

                uint32_t iterations{0};

                for (; iterations < MAX_ITERATIONS; iterations++)
                {
                    z = pow(z, 2) + constant;

                    if (std::norm(z) > 4)
                        break;
                }

                // Black background for the fractal and dynamic colours
                sf::Color color;

                if (iterations == MAX_ITERATIONS)
                    color = sf::Color::Black;
                else
                    color = palette.getColor(iterations);

                image.setPixel(x, y, color);
            }
        }
    };

    for (uint32_t i = 0; i < numThreads; ++i)
    {
        uint32_t startY = i * rowsPerThread;
        uint32_t endY = (i == numThreads - 1) ? HEIGHT : startY + rowsPerThread;
        threads.emplace_back(worker, startY, endY);
    }

    for (auto &t : threads)
        if (t.joinable())
            t.join();

}

void Fractals::run()
{
    // Create a window
    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Fractals");
    window.setFramerateLimit(30);

    // Create an image to store the fractal
    sf::Image image;
    image.create(WIDTH, HEIGHT, sf::Color::Black);

    // Create a texture & sprite to display the fractal
    sf::Texture texture;
    sf::Sprite sprite;

    // Create a font and text to display the iterations and coordinates
    // sf::Font font;
    // if (!font.loadFromFile("arial.ttf"))
    //     std::cout << "Error loading font!" << std::endl;

    // sf::Text text;
    // text.setFont(font);
    // text.setCharacterSize(20);
    // text.setFillColor(sf::Color::White);
    // text.setPosition(10, 10);

    // text.setString("Iterations: " + std::to_string(MAX_ITERATIONS) + "\n" +
    //                 "Re: [" + std::to_string(min_Re) + ", " + std::to_string(max_Re) + "]\n" +
    //                 "Im: [" + std::to_string(min_Im) + ", " + std::to_string(max_Im) + "]");
    
    // window.draw(text);

    // Calculate and plot Fractal once
    calculateFractal(image);
    texture.loadFromImage(image);
    sprite.setTexture(texture);

    // Save the fractal to a file
    srand(time(0));
    std::string filename = "E:/Fractals/imgs/fractal_" + std::to_string(rand()) + "_" + std::to_string(MAX_ITERATIONS) + ".jpg";

    // Display the fractal in the window
    while (window.isOpen())
    {
        sf::Event event;
        bool needsRecalculation = false;

        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();

            // Handle Zoom, Pan, Scroll
            if (event.type == sf::Event::KeyPressed)
            {
                // Delta for Panning
                double X = (max_Re - min_Re) * deltaX * 0.1;
                double Y = (max_Im - min_Im) * deltaY * 0.1;

                switch (event.key.code)
                {
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

                    case sf::Keyboard::Space:
                        image.saveToFile(filename);
                        break;
                    
                    case sf::Keyboard::Escape:
                        window.close();
                        break;
                        
                    default:
                        break;
                }
            }

            // Increase or decrease iterations
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
                auto zoom = [this](double z, double mouseX, double mouseY) {
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
            // Recalculate and plot Fractal
            calculateFractal(image);

            // Update the texture and sprite
            texture.loadFromImage(image);
            sprite.setTexture(texture);
        }

        window.clear();

        // Display coordinates and iterations as window title
        window.setTitle("Fractals - " + std::to_string(MAX_ITERATIONS) + " iterations" +
                        " - Re: [" + std::to_string(min_Re) + ", " + std::to_string(max_Re) + "]" +
                        " - Im: [" + std::to_string(min_Im) + ", " + std::to_string(max_Im) + "]");
        
        // window.draw(text);
        window.draw(sprite);
        
        window.display();
    }
}
