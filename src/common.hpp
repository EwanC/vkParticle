// Copyright (c) 2025 Ewan Crawford

#pragma once

import vulkan_hpp;
#define GLFW_INCLUDE_VULKAN // REQUIRED only for GLFW CreateWindowSurface.
#include <GLFW/glfw3.h>

#include <array>
#include <string>
#include <vector>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>

struct Particle {
  glm::vec2 position;
  glm::vec2 velocity;
  glm::vec4 color;

  static vk::VertexInputBindingDescription getBindingDescription() {
    return {0, sizeof(Particle), vk::VertexInputRate::eVertex};
  }

  static std::array<vk::VertexInputAttributeDescription, 2>
  getAttributeDescriptions() {
    return {vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Sfloat,
                                                offsetof(Particle, position)),
            vk::VertexInputAttributeDescription(1, 0,
                                                vk::Format::eR32G32B32A32Sfloat,
                                                offsetof(Particle, color))};
  }
};

struct UniformBufferObject {
  float deltaTime = 1.0f;
};

struct vkParticle {
  void run(); // called by main.cpp

  bool framebufferResized = false; // Accessed by GLFW callback

private:
  /*
   * Member methods
   */
  void initWindow();
  void initVulkan();
  void mainLoop();
  void cleanup();
  void createInstance();
  void pickPhysicalDevice();
  void createLogicalDevice();
  void setupDebugMessenger();
  void createSurface();
  void createSwapChain();
  void createImageViews();
  void createGraphicsPipeline();
  void createComputePipeline();
  void createCommandPool();
  void createCommandBuffers();
  void createComputeCommandBuffers();
  void createSyncObjects();
  void recordCommandBuffer(uint32_t imageIndex);
  void recordComputeCommandBuffer();
  void transition_image_layout(uint32_t imageIndex, vk::ImageLayout old_layout,
                               vk::ImageLayout new_layout,
                               vk::AccessFlags2 src_access_mask,
                               vk::AccessFlags2 dst_access_mask,
                               vk::PipelineStageFlags2 src_stage_mask,
                               vk::PipelineStageFlags2 dst_stage_mask);
  void drawFrame();
  void recreateSwapChain();
  void cleanupSwapChain();
  void createShaderStorageBuffers();
  void createUniformBuffers();
  void createDescriptorPool();
  void createComputeDescriptorSets();
  void copyBuffer(vk::raii::Buffer &srcBuffer, vk::raii::Buffer &dstBuffer,
                  vk::DeviceSize size);
  void createComputeDescriptorSetLayout();
  void updateUniformBuffer(uint32_t currentImage);

  /*
   * Member variables
   */

  GLFWwindow *MWindow = nullptr;
  vk::raii::Context MContext;
  vk::raii::Instance MInstance = nullptr;
  vk::raii::DebugUtilsMessengerEXT MDebugMessenger = nullptr;
  vk::raii::SurfaceKHR MSurface = nullptr;
  vk::raii::PhysicalDevice MPhysicalDevice = nullptr;
  vk::raii::Device MDevice = nullptr;
  uint32_t MQueueIndex = ~0;
  vk::raii::Queue MQueue = nullptr;
  vk::raii::SwapchainKHR MSwapChain = nullptr;
  std::vector<vk::Image> MSwapChainImages;
  vk::SurfaceFormatKHR MSwapChainSurfaceFormat;
  vk::Extent2D MSwapChainExtent;
  std::vector<vk::raii::ImageView> MSwapChainImageViews;

  vk::raii::PipelineLayout MPipelineLayout = nullptr;
  vk::raii::PipelineLayout MComputePipelineLayout = nullptr;
  vk::raii::Pipeline MGraphicsPipeline = nullptr;
  vk::raii::Pipeline MComputePipeline = nullptr;

  vk::raii::DescriptorSetLayout MComputeDescriptorSetLayout = nullptr;
  vk::raii::DescriptorPool MDescriptorPool = nullptr;
  std::vector<vk::raii::DescriptorSet> MComputeDescriptorSets;

  std::vector<vk::raii::Buffer> MShaderStorageBuffers;
  std::vector<vk::raii::DeviceMemory> MShaderStorageBuffersMemory;

  std::vector<vk::raii::Buffer> MUniformBuffers;
  std::vector<vk::raii::DeviceMemory> MUniformBuffersMemory;
  std::vector<void *> MUniformBuffersMapped;

  vk::raii::CommandPool MCommandPool = nullptr;
  std::vector<vk::raii::CommandBuffer> MCommandBuffers;
  std::vector<vk::raii::CommandBuffer> MComputeCommandBuffers;

  vk::raii::Semaphore MSemaphore = nullptr;
  uint64_t MTimelineValue = 0;
  std::vector<vk::raii::Fence> MInFlightFences;
  uint32_t MCurrentFrame = 0;

  double MLastFrameTime = 0.0;
  double MLastTime = 0.0;

  std::vector<const char *> MRequiredDeviceExtension = {
      vk::KHRSwapchainExtensionName,
      vk::KHRSpirv14ExtensionName,
      vk::KHRSynchronization2ExtensionName,
      vk::KHRCreateRenderpass2ExtensionName,
      vk::KHRShaderDrawParametersExtensionName,
  };

  /*
   * static members
   */
  static const uint32_t SWindowWidth = 800;
  static const uint32_t SWindowHeight = 600;
  static const unsigned SMaxFramesInFlight = 2;
  static const uint64_t SFenceTimeout = 100000000;
  static const uint32_t SParticleCount = 8192;
  static constexpr bool SEnableValidationLayers =
#ifdef NDEBUG
      false;
#else
      true;
#endif
  static const std::vector<const char *> SValidationLayers;
};

/*
 * Free functions from shader_file.cpp
 */
std::vector<char> readFile(const std::string &filename);

[[nodiscard]] vk::raii::ShaderModule
createShaderModule(const std::vector<char> &code, vk::raii::Device &device);
