// Copyright (c) 2025 Ewan Crawford

#pragma once

import vulkan_hpp;
#define GLFW_INCLUDE_VULKAN // REQUIRED only for GLFW CreateWindowSurface.
#include <GLFW/glfw3.h>

#include <array>
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
  void run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
  }

  // Accessed by GLFW callback
  bool framebufferResized = false;

private:
  void initWindow();
  void initVulkan();
  void mainLoop();
  void cleanup();
  void createInstance();
  std::vector<const char *> getRequiredExtensions() const;
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

  GLFWwindow *window = nullptr;
  vk::raii::Context context;
  vk::raii::Instance instance = nullptr;
  vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
  vk::raii::SurfaceKHR surface = nullptr;
  vk::raii::PhysicalDevice physicalDevice = nullptr;
  vk::raii::Device device = nullptr;
  uint32_t queueIndex = ~0;
  vk::raii::Queue queue = nullptr;
  vk::raii::SwapchainKHR swapChain = nullptr;
  std::vector<vk::Image> swapChainImages;
  vk::SurfaceFormatKHR swapChainSurfaceFormat;
  vk::Extent2D swapChainExtent;
  std::vector<vk::raii::ImageView> swapChainImageViews;

  vk::raii::PipelineLayout pipelineLayout = nullptr;
  vk::raii::Pipeline graphicsPipeline = nullptr;

  vk::raii::DescriptorSetLayout computeDescriptorSetLayout = nullptr;
  vk::raii::PipelineLayout computePipelineLayout = nullptr;
  vk::raii::Pipeline computePipeline = nullptr;

  std::vector<vk::raii::Buffer> shaderStorageBuffers;
  std::vector<vk::raii::DeviceMemory> shaderStorageBuffersMemory;

  std::vector<vk::raii::Buffer> uniformBuffers;
  std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
  std::vector<void *> uniformBuffersMapped;

  vk::raii::DescriptorPool descriptorPool = nullptr;
  std::vector<vk::raii::DescriptorSet> computeDescriptorSets;

  vk::raii::CommandPool commandPool = nullptr;
  std::vector<vk::raii::CommandBuffer> commandBuffers;
  std::vector<vk::raii::CommandBuffer> computeCommandBuffers;

  vk::raii::Semaphore semaphore = nullptr;
  uint64_t timelineValue = 0;
  std::vector<vk::raii::Fence> inFlightFences;
  uint32_t currentFrame = 0;

  double lastFrameTime = 0.0;
  double lastTime = 0.0;

  std::vector<const char *> requiredDeviceExtension = {
      vk::KHRSwapchainExtensionName,
      vk::KHRSpirv14ExtensionName,
      vk::KHRSynchronization2ExtensionName,
      vk::KHRCreateRenderpass2ExtensionName,
      vk::KHRShaderDrawParametersExtensionName,
  };

  // static members
  static const uint32_t windowWidth = 800;
  static const uint32_t windowHeight = 600;
  static const unsigned MaxFramesInFlight = 2;
  static const uint64_t FenceTimeout = 100000000;
  static const uint32_t ParticleCount = 8192;
  static constexpr bool enableValidationLayers =
#ifdef NDEBUG
      false;
#else
      true;
#endif
  static const std::vector<const char *> validationLayers;
};
