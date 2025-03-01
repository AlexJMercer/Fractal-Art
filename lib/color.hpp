#pragma once

#include <SFML/Graphics.hpp>

#include <iostream>
#include <vector>
#include <random>

class ColorPalette
{
public:
  std::vector<sf::Color> colors;

  ColorPalette();

  const sf::Color &getColor(std::size_t index) const;
  void generateRandomPalette();

  void generateGradientPalette();
};
