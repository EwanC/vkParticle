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

/// @brief Class used to interface with shader device code
struct Particle {
  glm::vec2 position;
  glm::vec2 velocity;
  glm::vec4 color;

  // Tells the runtime what stride to use for vertex data
  static vk::VertexInputBindingDescription getBindingDescription() {
    return {0, sizeof(Particle), vk::VertexInputRate::eVertex};
  }

  static std::array<vk::VertexInputAttributeDescription, 2>
  getAttributeDescriptions() {
    return {
        // In Vertex shader input, we have a float2 position struct attribute
        // followed by a float4 color attribute.
        vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Sfloat,
                                            offsetof(Particle, position)),
        vk::VertexInputAttributeDescription(
            1, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(Particle, color))};
  }
};

/// @brief uniform buffer used in compute shader
struct UniformBufferObject {
  float deltaTime = 1.0f;
};

/// @brief Class holding RAII state of the application
struct vkParticle {
  /// @brief User code entry-point, called by main.cpp
  void run();

  /// @brief Set by GLFW callback when window is resized.
  bool MFramebufferResized = false;

private:
  /*
   * Member methods
   */

  /// @brief Initializes GLFW and creates a window.
  void initWindow();
  /// @brief Creates required VK objects.
  void initVulkan();
  /// @brief Game loop that draws frames until user exists the program.
  void mainLoop();
  /// @brief Tears down GLFW instance on program exit.
  void cleanup();

  /// @brief Creates a VkInstance with required extensions and layers.
  void createInstance();
  /// @brief Registers validation layers, if enabled.
  void setupDebugMessenger();
  /// @brief Creates a VkSurfaceKHR window surface to interface with GLFW.
  void createSurface();
  /// @brief Selects a VkPhysicalDevice to use from the VK instance based on
  /// the application requirements.
  void pickPhysicalDevice();
  /// @brief Creates a logical device and queue with required capabilities.
  void createLogicalDevice();
  /// @brief Creates a swap chain image buffer for rendering.
  void createSwapChain();
  /// @brief Creates a view into each image in the swap chain.
  void createImageViews();
  /// @brief Defines descriptor set layout for compute shader of a uniform
  /// buffer and 2 storage buffers.
  void createComputeDescriptorSetLayout();
  /// @brief Loads vertex & fragment shaders,
  /// and creates graphics pipeline.
  void createGraphicsPipeline();
  /// @brief Loads compute shader, and creates compute pipeline.
  void createComputePipeline();
  /// @brief Creates a command pool.
  void createCommandPool();
  /// @brief Creates buffer for every frame of `Particle` objects copied to
  /// GPU-only memory from host-visible staging memory.
  void createShaderStorageBuffers();
  /// @brief Creates a persistently mapped uniformed buffer for every frame.
  void createUniformBuffers();
  /// @brief Defines descriptor pool for creating uniform and storage buffer
  /// descriptors from
  void createDescriptorPool();
  /// @brief For every frame, writes a descriptor set of a uniform buffer with
  /// the new time delta, as well as 2 storage buffers for the last and
  /// current frame particle position.
  void createComputeDescriptorSets();
  /// @brief Creates a command-buffer to use for graphics commands for each of
  /// the possible frames in flight.
  void createGraphicsCommandBuffers();
  /// @brief Creates a command-buffer to use for compute commands for each of
  /// the possible frames in flight.
  void createComputeCommandBuffers();
  /// @brief Creates timeline semaphore and fences for synchronization.
  void createSyncObjects();

  /// @brief Add commands to graphics command-buffer
  /// @param[in] imageIndex Index in swap chain of current image for frame.
  void recordGraphicsCommandBuffer(uint32_t imageIndex);
  /// @brief Add commands to compute command-buffer
  void recordComputeCommandBuffer();
  /// @brief Submits the command-buffers to the queue,
  /// and presents the new frame.
  void drawFrame();
  /// @brief Reconfigures the swap chain image formats if the window is resized.
  void recreateSwapChain();
  /// @brief Resets swap chain state.
  void cleanupSwapChain();
  /// @brief Adds a copy buffer command to a newly created one time submit
  /// command-buffer, and submits it to the queue with a blocking host wait.
  /// @param[in] srcBuffer Buffer to copy from.
  /// @param[in] dstBuffer Buffer to copy to.
  /// @param[in] Size Number of bytes to copy.
  void copyBuffer(vk::raii::Buffer &srcBuffer, vk::raii::Buffer &dstBuffer,
                  vk::DeviceSize size);
  /// @brief Sets the uniform buffer object data to the latest time delta.
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
  std::vector<vk::raii::CommandBuffer> MGraphicsCommandBuffers;
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
   * Static member variables
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

/// @brief Loads a file from disk.
/// @param[in] filename Path on disk for file to read.
/// @returns Vector of char bytes with file contents.
std::vector<char> readFile(const std::string &filename);

/// @brief Creates a VK shader module from shader source
/// @param[in] code Source code of shader.
/// @param[in] device The device to created the model for.
/// @returns The created shader module.
[[nodiscard]] vk::raii::ShaderModule
createShaderModule(const std::vector<char> &code, vk::raii::Device &device);
