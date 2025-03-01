#include "../lib/color.hpp"

ColorPalette::ColorPalette() {}

const sf::Color &ColorPalette::getColor(std::size_t index) const
{
  return colors[index % colors.size()];
}

void ColorPalette::generateRandomPalette()
{
  colors.clear();
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<> dis(0, 255);

  for (int i = 0; i < 256; ++i)
  {
    colors.emplace_back(dis(gen), dis(gen), dis(gen));
  }
}

void ColorPalette::generateGradientPalette()
{
  colors.clear();

  // Randomize the start and end colors
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(100, 255);
  std::uniform_int_distribution<> dis2(0, 35);

  // sf::Color startColor{0, 0, 0};
  sf::Color startColor{static_cast<sf::Uint8>(dis2(gen)), static_cast<sf::Uint8>(dis2(gen)), static_cast<sf::Uint8>(dis2(gen))};
  sf::Color endColor{static_cast<sf::Uint8>(dis(gen)), static_cast<sf::Uint8>(dis(gen)), static_cast<sf::Uint8>(dis(gen))};

  std::cout << "Start Color: " << static_cast<int>(startColor.r) << ", " << static_cast<int>(startColor.g) << ", " << static_cast<int>(startColor.b) << std::endl;
  std::cout << "End Color: " << static_cast<int>(endColor.r) << ", " << static_cast<int>(endColor.g) << ", " << static_cast<int>(endColor.b) << std::endl;

  for (std::size_t i = 0; i < 256; ++i)
  {
    float t = static_cast<float>(i) / (256);

    sf::Uint8 r = static_cast<sf::Uint8>(
        startColor.r + t * (endColor.r - startColor.r));

    sf::Uint8 g = static_cast<sf::Uint8>(
        startColor.g + t * (endColor.g - startColor.g));

    sf::Uint8 b = static_cast<sf::Uint8>(
        startColor.b + t * (endColor.b - startColor.b));

    colors.emplace_back(r, g, b);
  }
}