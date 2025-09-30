// Copyright (c) 2025 Ewan Crawford

#include "vk_particle.hpp"
#include <iostream>

int main() {
  vkParticle app;

  try {
    app.run();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return -1;
  }

  return 0;
}
