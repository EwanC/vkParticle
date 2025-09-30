// Copyright (c) 2025 Ewan Crawford

#pragma once

import vulkan_hpp;

#include <string>
#include <vector>

std::vector<char> readFile(const std::string &filename);

[[nodiscard]] vk::raii::ShaderModule
createShaderModule(const std::vector<char> &code, vk::raii::Device &device);
