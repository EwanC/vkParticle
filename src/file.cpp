// Copyright (c) 2025 Ewan Crawford

#include "file.hpp"
#include <cstdint>
#include <format>
#include <fstream>

std::vector<char> readFile(const std::string &filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error(std::format("failed to open file {}", filename));
  }
  std::vector<char> buffer(file.tellg());
  file.seekg(0, std::ios::beg);
  file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
  file.close();
  return buffer;
}

[[nodiscard]] vk::raii::ShaderModule
createShaderModule(const std::vector<char> &code, vk::raii::Device &device) {
  vk::ShaderModuleCreateInfo createInfo{
      .codeSize = code.size() * sizeof(char),
      .pCode = reinterpret_cast<const uint32_t *>(code.data())};
  vk::raii::ShaderModule shaderModule{device, createInfo};

  return shaderModule;
}
