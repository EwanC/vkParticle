// Copyright (c) 2025-2026 Ewan Crawford

#include "common.hpp"
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
