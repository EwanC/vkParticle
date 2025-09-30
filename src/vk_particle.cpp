// Copyright (c) 2025 Ewan Crawford

#include "vk_particle.hpp"
#include "file.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <random>
#include <stdexcept>

// For macros not exported by C++ module
#include <vulkan/vulkan_core.h>

const std::vector<const char *> vkParticle::validationLayers = {
    "VK_LAYER_KHRONOS_validation"};

namespace {
void framebufferResizeCallback(GLFWwindow *window, int width, int height) {
  auto app = reinterpret_cast<vkParticle *>(glfwGetWindowUserPointer(window));
  app->framebufferResized = true;
}
} // anonymous namespace
void vkParticle::initWindow() {
  glfwInit();

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

  window = glfwCreateWindow(windowWidth, windowHeight, "vkParticle", nullptr,
                            nullptr);
  glfwSetWindowUserPointer(window, this);
  glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
}

void vkParticle::initVulkan() {
  createInstance();
  setupDebugMessenger();
  createSurface();
  pickPhysicalDevice();
  createLogicalDevice();
  createSwapChain();
  createImageViews();
  createComputeDescriptorSetLayout();
  createGraphicsPipeline();
  createComputePipeline();
  createCommandPool();
  createShaderStorageBuffers();
  createUniformBuffers();
  createDescriptorPool();
  createComputeDescriptorSets();
  createCommandBuffers();
  createComputeCommandBuffers();
  createSyncObjects();
}

void vkParticle::mainLoop() {
  while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
         !glfwWindowShouldClose(window)) {
    glfwPollEvents();
    drawFrame();
    // We want to animate the particle system using the last frames time to get
    // smooth, frame-rate independent animation
    double currentTime = glfwGetTime();
    lastFrameTime = (currentTime - lastTime) * 1000.0;
    lastTime = currentTime;
  }
  device.waitIdle();
}

void vkParticle::cleanup() {
  glfwDestroyWindow(window);
  glfwTerminate();
}

void vkParticle::createInstance() {
  constexpr vk::ApplicationInfo appInfo{
      .pApplicationName = "vkParticle",
      .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
      .pEngineName = "No Engine",
      .engineVersion = VK_MAKE_VERSION(1, 0, 0),
      .apiVersion = vk::ApiVersion14};

  // Get the required layers
  std::vector<char const *> requiredLayers;
  if (enableValidationLayers) {
    requiredLayers.assign(validationLayers.begin(), validationLayers.end());
  }

  auto layerProperties = context.enumerateInstanceLayerProperties();
  for (auto const &requiredLayer : requiredLayers) {
    bool layerUnsupported = std::ranges::none_of(
        layerProperties, [requiredLayer](auto const &layerProperty) {
          return strcmp(layerProperty.layerName, requiredLayer) == 0;
        });
    if (layerUnsupported) {
      throw std::runtime_error("Required layer not supported: " +
                               std::string(requiredLayer));
    }
  }

  // Get required extensions and check all are supported by the Vulkan
  // implementation.
  auto requiredExtensions = getRequiredExtensions();
  auto extensionProperties = context.enumerateInstanceExtensionProperties();
  for (auto const &requiredExtension : requiredExtensions) {
    bool extUnsupported = std::ranges::none_of(
        extensionProperties,
        [requiredExtension](auto const &extensionProperty) {
          return strcmp(extensionProperty.extensionName, requiredExtension) ==
                 0;
        });
    if (extUnsupported) {
      throw std::runtime_error("Required extension not supported: " +
                               std::string(requiredExtension));
    }
  }

  vk::InstanceCreateInfo createInfo{
      .pApplicationInfo = &appInfo,
      .enabledLayerCount = static_cast<uint32_t>(requiredLayers.size()),
      .ppEnabledLayerNames = requiredLayers.data(),
      .enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size()),
      .ppEnabledExtensionNames = requiredExtensions.data()};
  instance = vk::raii::Instance(context, createInfo);
}

std::vector<const char *> vkParticle::getRequiredExtensions() const {
  uint32_t glfwExtCount = 0;
  auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtCount);

  std::vector extensions(glfwExtensions, glfwExtensions + glfwExtCount);
  if (enableValidationLayers) {
    extensions.push_back(vk::EXTDebugUtilsExtensionName);
  }
  return extensions;
}

void vkParticle::pickPhysicalDevice() {
  std::vector<vk::raii::PhysicalDevice> devices =
      instance.enumeratePhysicalDevices();
  const auto devIter = std::ranges::find_if(devices, [&](auto const &device) {
    // Check if the device supports the Vulkan 1.3 API version
    bool supportsVulkan1_3 =
        device.getProperties().apiVersion >= VK_API_VERSION_1_3;

    // Check if any of the queue families support graphics operations
    auto queueFamilies = device.getQueueFamilyProperties();
    bool supportsGraphics =
        std::ranges::any_of(queueFamilies, [](auto const &qfp) {
          return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics);
        });

    // Check if all required device extensions are available
    auto availableDeviceExtensions =
        device.enumerateDeviceExtensionProperties();
    bool supportsAllRequiredExtensions = std::ranges::all_of(
        requiredDeviceExtension,
        [&availableDeviceExtensions](auto const &requiredDeviceExtension) {
          return std::ranges::any_of(
              availableDeviceExtensions,
              [requiredDeviceExtension](auto const &availableDeviceExtension) {
                return strcmp(availableDeviceExtension.extensionName,
                              requiredDeviceExtension) == 0;
              });
        });

    auto features = device.template getFeatures2<
        vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features,
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
        vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR>();
    bool supportsRequiredFeatures =
        features.template get<vk::PhysicalDeviceVulkan13Features>()
            .dynamicRendering &&
        features.template get<vk::PhysicalDeviceVulkan13Features>()
            .synchronization2 &&
        features
            .template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>()
            .extendedDynamicState &&
        features.template get<vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR>()
            .timelineSemaphore;

    return supportsVulkan1_3 && supportsGraphics &&
           supportsAllRequiredExtensions && supportsRequiredFeatures;
  });
  if (devIter != devices.end()) {
    physicalDevice = *devIter;
  } else {
    throw std::runtime_error("failed to find a suitable GPU!");
  }
}

void vkParticle::createLogicalDevice() {
  // find the index of the first queue family that supports graphics
  std::vector<vk::QueueFamilyProperties> queueFamilyProperties =
      physicalDevice.getQueueFamilyProperties();

  // get the first index into queueFamilyProperties which supports both graphics
  // and present
  for (uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size();
       qfpIndex++) {
    if ((queueFamilyProperties[qfpIndex].queueFlags &
         vk::QueueFlagBits::eGraphics) &&
        (queueFamilyProperties[qfpIndex].queueFlags &
         vk::QueueFlagBits::eCompute) &&
        physicalDevice.getSurfaceSupportKHR(qfpIndex, *surface)) {
      // found a queue family that supports both graphics and present
      queueIndex = qfpIndex;
      break;
    }
  }
  if (queueIndex == ~0) {
    throw std::runtime_error(
        "Could not find a queue for graphics and present -> terminating");
  }

  // query for Vulkan 1.3 features
  vk::StructureChain<vk::PhysicalDeviceFeatures2,
                     vk::PhysicalDeviceVulkan13Features,
                     vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
                     vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR>
      featureChain = {
          {}, // vk::PhysicalDeviceFeatures2
          {.synchronization2 = true,
           .dynamicRendering = true}, // vk::PhysicalDeviceVulkan13Features
          {.extendedDynamicState =
               true}, // vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT
          {.timelineSemaphore = true} // vk::PhysicalDeviceTimelineSemaphoreKHR
      };

  // create a Device
  float queuePriority = 0.0f;
  vk::DeviceQueueCreateInfo deviceQueueCreateInfo{
      .queueFamilyIndex = queueIndex,
      .queueCount = 1,
      .pQueuePriorities = &queuePriority};
  vk::DeviceCreateInfo deviceCreateInfo{
      .pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &deviceQueueCreateInfo,
      .enabledExtensionCount =
          static_cast<uint32_t>(requiredDeviceExtension.size()),
      .ppEnabledExtensionNames = requiredDeviceExtension.data()};

  device = vk::raii::Device(physicalDevice, deviceCreateInfo);
  queue = vk::raii::Queue(device, queueIndex, 0);
}

static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(
    vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
    vk::DebugUtilsMessageTypeFlagsEXT type,
    const vk::DebugUtilsMessengerCallbackDataEXT *pCallbackData, void *) {
  if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError ||
      severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning) {
    std::cerr << "validation layer: type " << to_string(type)
              << " msg: " << pCallbackData->pMessage << std::endl;
  }

  return vk::False;
}

void vkParticle::setupDebugMessenger() {
  if (!enableValidationLayers)
    return;

  vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
  vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(
      vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
      vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
      vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
  vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT{
      .messageSeverity = severityFlags,
      .messageType = messageTypeFlags,
      .pfnUserCallback = &debugCallback};

  debugMessenger =
      instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
}

void vkParticle::createSurface() {
  // GLFW only deals with C API
  VkSurfaceKHR vkSurface;
  if (glfwCreateWindowSurface(*instance, window, nullptr, &vkSurface) != 0) {
    throw std::runtime_error("failed to create window surface!");
  }
  surface = vk::raii::SurfaceKHR(instance, vkSurface);
}

namespace {
uint32_t
chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const &surfaceCapabilities) {
  auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
  if ((0 < surfaceCapabilities.maxImageCount) &&
      (surfaceCapabilities.maxImageCount < minImageCount)) {
    minImageCount = surfaceCapabilities.maxImageCount;
  }
  return minImageCount;
}

vk::SurfaceFormatKHR chooseSwapSurfaceFormat(
    std::vector<vk::SurfaceFormatKHR> const &availableFormats) {
  assert(!availableFormats.empty());
  const auto formatIt =
      std::ranges::find_if(availableFormats, [](const auto &format) {
        return format.format == vk::Format::eB8G8R8A8Srgb &&
               format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
      });
  return formatIt != availableFormats.end() ? *formatIt : availableFormats[0];
}

vk::PresentModeKHR chooseSwapPresentMode(
    const std::vector<vk::PresentModeKHR> &availablePresentModes) {
  assert(std::ranges::any_of(availablePresentModes, [](auto presentMode) {
    return presentMode == vk::PresentModeKHR::eFifo;
  }));
  return std::ranges::any_of(availablePresentModes,
                             [](const vk::PresentModeKHR value) {
                               return vk::PresentModeKHR::eMailbox == value;
                             })
             ? vk::PresentModeKHR::eMailbox
             : vk::PresentModeKHR::eFifo;
}

vk::Extent2D chooseSwapExtent(GLFWwindow *window,
                              const vk::SurfaceCapabilitiesKHR &capabilities) {
  if (capabilities.currentExtent.width != 0xFFFFFFFF) {
    return capabilities.currentExtent;
  }
  int width, height;
  glfwGetFramebufferSize(window, &width, &height);

  return {std::clamp<uint32_t>(width, capabilities.minImageExtent.width,
                               capabilities.maxImageExtent.width),
          std::clamp<uint32_t>(height, capabilities.minImageExtent.height,
                               capabilities.maxImageExtent.height)};
}

} // end anonymous namespace

void vkParticle::createSwapChain() {
  auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
  swapChainExtent = chooseSwapExtent(window, surfaceCapabilities);
  swapChainSurfaceFormat =
      chooseSwapSurfaceFormat(physicalDevice.getSurfaceFormatsKHR(*surface));
  vk::SwapchainCreateInfoKHR swapChainCreateInfo{
      .surface = *surface,
      .minImageCount = chooseSwapMinImageCount(surfaceCapabilities),
      .imageFormat = swapChainSurfaceFormat.format,
      .imageColorSpace = swapChainSurfaceFormat.colorSpace,
      .imageExtent = swapChainExtent,
      .imageArrayLayers = 1,
      .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
      .imageSharingMode = vk::SharingMode::eExclusive,
      .preTransform = surfaceCapabilities.currentTransform,
      .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
      .presentMode = chooseSwapPresentMode(
          physicalDevice.getSurfacePresentModesKHR(*surface)),
      .clipped = true};

  swapChain = vk::raii::SwapchainKHR(device, swapChainCreateInfo);
  swapChainImages = swapChain.getImages();
}

void vkParticle::createImageViews() {
  assert(swapChainImageViews.empty());

  vk::ImageViewCreateInfo imageViewCreateInfo{
      .viewType = vk::ImageViewType::e2D,
      .format = swapChainSurfaceFormat.format,
      .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};
  for (auto image : swapChainImages) {
    imageViewCreateInfo.image = image;
    swapChainImageViews.emplace_back(device, imageViewCreateInfo);
  }
}

void vkParticle::createGraphicsPipeline() {
  // Shader setup
  vk::raii::ShaderModule shaderModule =
      createShaderModule(readFile("slang.spv"), device);

  vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
      .stage = vk::ShaderStageFlagBits::eVertex,
      .module = shaderModule,
      .pName = "vertMain"};
  vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
      .stage = vk::ShaderStageFlagBits::eFragment,
      .module = shaderModule,
      .pName = "fragMain"};
  vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo,
                                                      fragShaderStageInfo};

  auto bindingDescription = Particle::getBindingDescription();
  auto attributeDescriptions = Particle::getAttributeDescriptions();
  vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &bindingDescription,
      .vertexAttributeDescriptionCount =
          static_cast<uint32_t>(attributeDescriptions.size()),
      .pVertexAttributeDescriptions = attributeDescriptions.data()};

  vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
      .topology = vk::PrimitiveTopology::ePointList,
      .primitiveRestartEnable = vk::False};
  vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1,
                                                    .scissorCount = 1};
  vk::PipelineRasterizationStateCreateInfo rasterizer{
      .depthClampEnable = vk::False,
      .rasterizerDiscardEnable = vk::False,
      .polygonMode = vk::PolygonMode::eFill,
      .cullMode = vk::CullModeFlagBits::eBack,
      .frontFace = vk::FrontFace::eCounterClockwise,
      .depthBiasEnable = vk::False,
      .depthBiasSlopeFactor = 1.0f,
      .lineWidth = 1.0f};

  vk::PipelineMultisampleStateCreateInfo multisampling{
      .rasterizationSamples = vk::SampleCountFlagBits::e1,
      .sampleShadingEnable = vk::False};

  vk::PipelineColorBlendAttachmentState colorBlendAttachment{
      .blendEnable = vk::True,
      .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
      .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
      .colorBlendOp = vk::BlendOp::eAdd,
      .srcAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
      .dstAlphaBlendFactor = vk::BlendFactor::eZero,
      .alphaBlendOp = vk::BlendOp::eAdd,
      .colorWriteMask =
          vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
  };

  vk::PipelineColorBlendStateCreateInfo colorBlending{
      .logicOpEnable = vk::False,
      .logicOp = vk::LogicOp::eCopy,
      .attachmentCount = 1,
      .pAttachments = &colorBlendAttachment};

  // Can be changed without recreating the pipeline
  std::vector dynamicStates = {vk::DynamicState::eViewport,
                               vk::DynamicState::eScissor};
  vk::PipelineDynamicStateCreateInfo dynamicState{
      .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
      .pDynamicStates = dynamicStates.data()};

  vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
  pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

  vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo{
      .colorAttachmentCount = 1,
      .pColorAttachmentFormats = &swapChainSurfaceFormat.format};
  vk::GraphicsPipelineCreateInfo pipelineInfo{
      .pNext = &pipelineRenderingCreateInfo,
      .stageCount = 2,
      .pStages = shaderStages,
      .pVertexInputState = &vertexInputInfo,
      .pInputAssemblyState = &inputAssembly,
      .pViewportState = &viewportState,
      .pRasterizationState = &rasterizer,
      .pMultisampleState = &multisampling,
      .pColorBlendState = &colorBlending,
      .pDynamicState = &dynamicState,
      .layout = *pipelineLayout,
      .renderPass = nullptr};

  graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
}

void vkParticle::createCommandPool() {
  vk::CommandPoolCreateInfo poolInfo{
      .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
      .queueFamilyIndex = queueIndex};
  commandPool = vk::raii::CommandPool(device, poolInfo);
}
void vkParticle::createCommandBuffers() {
  commandBuffers.clear();
  vk::CommandBufferAllocateInfo allocInfo{
      .commandPool = commandPool,
      .level = vk::CommandBufferLevel::ePrimary,
      .commandBufferCount = MaxFramesInFlight};
  commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
}

void vkParticle::createSyncObjects() {
  inFlightFences.clear();

  vk::SemaphoreTypeCreateInfo semaphoreType{
      .semaphoreType = vk::SemaphoreType::eTimeline, .initialValue = 0};
  semaphore = vk::raii::Semaphore(device, {.pNext = &semaphoreType});
  timelineValue = 0;

  for (size_t i = 0; i < MaxFramesInFlight; i++) {
    vk::FenceCreateInfo fenceInfo{};
    inFlightFences.emplace_back(device, fenceInfo);
  }
}

void vkParticle::transition_image_layout(
    uint32_t imageIndex, vk::ImageLayout old_layout, vk::ImageLayout new_layout,
    vk::AccessFlags2 src_access_mask, vk::AccessFlags2 dst_access_mask,
    vk::PipelineStageFlags2 src_stage_mask,
    vk::PipelineStageFlags2 dst_stage_mask) {
  vk::ImageMemoryBarrier2 barrier = {
      .srcStageMask = src_stage_mask,
      .srcAccessMask = src_access_mask,
      .dstStageMask = dst_stage_mask,
      .dstAccessMask = dst_access_mask,
      .oldLayout = old_layout,
      .newLayout = new_layout,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = swapChainImages[imageIndex],
      .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                           .baseMipLevel = 0,
                           .levelCount = 1,
                           .baseArrayLayer = 0,
                           .layerCount = 1}};
  vk::DependencyInfo dependency_info = {.dependencyFlags = {},
                                        .imageMemoryBarrierCount = 1,
                                        .pImageMemoryBarriers = &barrier};
  commandBuffers[currentFrame].pipelineBarrier2(dependency_info);
}

void vkParticle::recordCommandBuffer(uint32_t imageIndex) {
  commandBuffers[currentFrame].reset();
  commandBuffers[currentFrame].begin({});
  // Before starting rendering, transition the swapchain image to
  // COLOR_ATTACHMENT_OPTIMAL
  transition_image_layout(
      imageIndex, vk::ImageLayout::eUndefined,
      vk::ImageLayout::eColorAttachmentOptimal,
      {}, // srcAccessMask (no need to wait for previous operations)
      vk::AccessFlagBits2::eColorAttachmentWrite,        // dstAccessMask
      vk::PipelineStageFlagBits2::eTopOfPipe,            // srcStage
      vk::PipelineStageFlagBits2::eColorAttachmentOutput // dstStage
  );
  vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
  vk::RenderingAttachmentInfo attachmentInfo = {
      .imageView = swapChainImageViews[imageIndex],
      .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
      .loadOp = vk::AttachmentLoadOp::eClear,
      .storeOp = vk::AttachmentStoreOp::eStore,
      .clearValue = clearColor};
  vk::RenderingInfo renderingInfo = {
      .renderArea = {.offset = {0, 0}, .extent = swapChainExtent},
      .layerCount = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments = &attachmentInfo};

  commandBuffers[currentFrame].beginRendering(renderingInfo);
  commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics,
                                            *graphicsPipeline);
  commandBuffers[currentFrame].setViewport(
      0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width),
                      static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
  commandBuffers[currentFrame].setScissor(
      0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
  commandBuffers[currentFrame].bindVertexBuffers(
      0, {shaderStorageBuffers[currentFrame]}, {0});
  commandBuffers[currentFrame].draw(ParticleCount, 1, 0, 0);
  commandBuffers[currentFrame].endRendering();
  // After rendering, transition the swapchain image to PRESENT_SRC
  transition_image_layout(
      imageIndex, vk::ImageLayout::eColorAttachmentOptimal,
      vk::ImageLayout::ePresentSrcKHR,
      vk::AccessFlagBits2::eColorAttachmentWrite,         // srcAccessMask
      {},                                                 // dstAccessMask
      vk::PipelineStageFlagBits2::eColorAttachmentOutput, // srcStage
      vk::PipelineStageFlagBits2::eBottomOfPipe           // dstStage
  );
  commandBuffers[currentFrame].end();
}

void vkParticle::updateUniformBuffer(uint32_t currentImage) {
  UniformBufferObject ubo{};
  ubo.deltaTime = static_cast<float>(lastFrameTime) * 2.f;
  memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
}

void vkParticle::drawFrame() {
  auto [result, imageIndex] = swapChain.acquireNextImage(
      UINT64_MAX, nullptr, *inFlightFences[currentFrame]);
  while (vk::Result::eTimeout ==
         device.waitForFences(*inFlightFences[currentFrame], vk::True,
                              UINT64_MAX)) {
    ;
  }
  device.resetFences(*inFlightFences[currentFrame]);

  // Update timeline value for this frame
  uint64_t computeWaitValue = timelineValue;
  uint64_t computeSignalValue = ++timelineValue;
  uint64_t graphicsWaitValue = computeSignalValue;
  uint64_t graphicsSignalValue = ++timelineValue;

  updateUniformBuffer(currentFrame);

  {
    recordComputeCommandBuffer();
    // Submit compute work
    vk::TimelineSemaphoreSubmitInfo computeTimelineInfo{
        .waitSemaphoreValueCount = 1,
        .pWaitSemaphoreValues = &computeWaitValue,
        .signalSemaphoreValueCount = 1,
        .pSignalSemaphoreValues = &computeSignalValue};

    vk::PipelineStageFlags waitStages[] = {
        vk::PipelineStageFlagBits::eComputeShader};

    vk::SubmitInfo computeSubmitInfo{.pNext = &computeTimelineInfo,
                                     .waitSemaphoreCount = 1,
                                     .pWaitSemaphores = &*semaphore,
                                     .pWaitDstStageMask = waitStages,
                                     .commandBufferCount = 1,
                                     .pCommandBuffers =
                                         &*computeCommandBuffers[currentFrame],
                                     .signalSemaphoreCount = 1,
                                     .pSignalSemaphores = &*semaphore};

    queue.submit(computeSubmitInfo, nullptr);
  }
  {
    recordCommandBuffer(imageIndex);

    // Submit graphics work (waits for compute to finish)
    vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eVertexInput;
    vk::TimelineSemaphoreSubmitInfo graphicsTimelineInfo{
        .waitSemaphoreValueCount = 1,
        .pWaitSemaphoreValues = &graphicsWaitValue,
        .signalSemaphoreValueCount = 1,
        .pSignalSemaphoreValues = &graphicsSignalValue};

    vk::SubmitInfo graphicsSubmitInfo{.pNext = &graphicsTimelineInfo,
                                      .waitSemaphoreCount = 1,
                                      .pWaitSemaphores = &*semaphore,
                                      .pWaitDstStageMask = &waitStage,
                                      .commandBufferCount = 1,
                                      .pCommandBuffers =
                                          &*commandBuffers[currentFrame],
                                      .signalSemaphoreCount = 1,
                                      .pSignalSemaphores = &*semaphore};

    queue.submit(graphicsSubmitInfo, nullptr);

    // Present the image (wait for graphics to finish)
    vk::SemaphoreWaitInfo waitInfo{.semaphoreCount = 1,
                                   .pSemaphores = &*semaphore,
                                   .pValues = &graphicsSignalValue};

    // Wait for graphics to complete before presenting
    while (vk::Result::eTimeout == device.waitSemaphores(waitInfo, UINT64_MAX))
      ;

    vk::PresentInfoKHR presentInfo{.waitSemaphoreCount =
                                       0, // No binary semaphores needed
                                   .pWaitSemaphores = nullptr,
                                   .swapchainCount = 1,
                                   .pSwapchains = &*swapChain,
                                   .pImageIndices = &imageIndex};

    result = queue.presentKHR(presentInfo);
    if (result == vk::Result::eErrorOutOfDateKHR ||
        result == vk::Result::eSuboptimalKHR || framebufferResized) {
      framebufferResized = false;
      recreateSwapChain();
    } else if (result != vk::Result::eSuccess) {
      throw std::runtime_error("failed to present swap chain image!");
    }
  }

  currentFrame = (currentFrame + 1) % MaxFramesInFlight;
}

void vkParticle::recreateSwapChain() {
  int width = 0, height = 0;
  glfwGetFramebufferSize(window, &width, &height);
  while (width == 0 || height == 0) {
    glfwGetFramebufferSize(window, &width, &height);
    glfwWaitEvents();
  }

  device.waitIdle();

  cleanupSwapChain();
  createSwapChain();
  createImageViews();
}

void vkParticle::cleanupSwapChain() {
  swapChainImageViews.clear();
  swapChain = nullptr;
}

namespace {
uint32_t findMemoryType(vk::raii::PhysicalDevice &physicalDevice,
                        uint32_t typeFilter,
                        vk::MemoryPropertyFlags properties) {
  vk::PhysicalDeviceMemoryProperties memProperties =
      physicalDevice.getMemoryProperties();

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags &
                                    properties) == properties) {
      return i;
    }
  }

  throw std::runtime_error("failed to find suitable memory type!");
}

void createBuffer(vk::raii::Device &device,
                  vk::raii::PhysicalDevice &physicalDevice, vk::DeviceSize size,
                  vk::BufferUsageFlags usage,
                  vk::MemoryPropertyFlags properties, vk::raii::Buffer &buffer,
                  vk::raii::DeviceMemory &bufferMemory) {
  vk::BufferCreateInfo bufferInfo{
      .size = size, .usage = usage, .sharingMode = vk::SharingMode::eExclusive};
  buffer = vk::raii::Buffer(device, bufferInfo);
  vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
  vk::MemoryAllocateInfo allocInfo{
      .allocationSize = memRequirements.size,
      .memoryTypeIndex = findMemoryType(
          physicalDevice, memRequirements.memoryTypeBits, properties)};
  bufferMemory = vk::raii::DeviceMemory(device, allocInfo);
  buffer.bindMemory(bufferMemory, 0);
}
} // namespace

void vkParticle::copyBuffer(vk::raii::Buffer &srcBuffer,
                            vk::raii::Buffer &dstBuffer, vk::DeviceSize size) {
  vk::CommandBufferAllocateInfo allocInfo{.commandPool = commandPool,
                                          .level =
                                              vk::CommandBufferLevel::ePrimary,
                                          .commandBufferCount = 1};
  vk::raii::CommandBuffer commandCopyBuffer =
      std::move(device.allocateCommandBuffers(allocInfo).front());
  commandCopyBuffer.begin(vk::CommandBufferBeginInfo{
      .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  commandCopyBuffer.copyBuffer(*srcBuffer, *dstBuffer,
                               vk::BufferCopy(0, 0, size));
  commandCopyBuffer.end();
  queue.submit(vk::SubmitInfo{.commandBufferCount = 1,
                              .pCommandBuffers = &*commandCopyBuffer},
               nullptr);
  queue.waitIdle();
}

void vkParticle::createShaderStorageBuffers() {
  // Initialize particles
  std::default_random_engine rndEngine(static_cast<unsigned>(time(nullptr)));
  std::uniform_real_distribution rndDist(0.0f, 1.0f);

  // Initial particle positions on a circle
  std::vector<Particle> particles(ParticleCount);
  for (auto &particle : particles) {
    float r = 0.25f * sqrtf(rndDist(rndEngine));
    float theta = rndDist(rndEngine) * 2.0f * 3.14159265358979323846f;
    float x = r * cosf(theta) * windowHeight / windowWidth;
    float y = r * sinf(theta);
    particle.position = glm::vec2(x, y);
    particle.velocity = normalize(glm::vec2(x, y)) * 0.00025f;
    particle.color = glm::vec4(rndDist(rndEngine), rndDist(rndEngine),
                               rndDist(rndEngine), 1.0f);
  }

  vk::DeviceSize bufferSize = sizeof(Particle) * ParticleCount;

  // Create a staging buffer used to upload data to the gpu
  vk::raii::Buffer stagingBuffer({});
  vk::raii::DeviceMemory stagingBufferMemory({});
  createBuffer(device, physicalDevice, bufferSize,
               vk::BufferUsageFlagBits::eTransferSrc,
               vk::MemoryPropertyFlagBits::eHostVisible |
                   vk::MemoryPropertyFlagBits::eHostCoherent,
               stagingBuffer, stagingBufferMemory);

  void *dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
  memcpy(dataStaging, particles.data(), (size_t)bufferSize);
  stagingBufferMemory.unmapMemory();

  shaderStorageBuffers.clear();
  shaderStorageBuffersMemory.clear();

  // Copy initial particle data to all storage buffers
  for (size_t i = 0; i < MaxFramesInFlight; i++) {
    vk::raii::Buffer shaderStorageBufferTemp({});
    vk::raii::DeviceMemory shaderStorageBufferTempMemory({});
    createBuffer(device, physicalDevice, bufferSize,
                 vk::BufferUsageFlagBits::eStorageBuffer |
                     vk::BufferUsageFlagBits::eVertexBuffer |
                     vk::BufferUsageFlagBits::eTransferDst,
                 vk::MemoryPropertyFlagBits::eDeviceLocal,
                 shaderStorageBufferTemp, shaderStorageBufferTempMemory);
    copyBuffer(stagingBuffer, shaderStorageBufferTemp, bufferSize);
    shaderStorageBuffers.emplace_back(std::move(shaderStorageBufferTemp));
    shaderStorageBuffersMemory.emplace_back(
        std::move(shaderStorageBufferTempMemory));
  }
}

void vkParticle::createUniformBuffers() {
  uniformBuffers.clear();
  uniformBuffersMemory.clear();
  uniformBuffersMapped.clear();

  for (size_t i = 0; i < MaxFramesInFlight; i++) {
    vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
    vk::raii::Buffer buffer({});
    vk::raii::DeviceMemory bufferMem({});
    createBuffer(device, physicalDevice, bufferSize,
                 vk::BufferUsageFlagBits::eUniformBuffer,
                 vk::MemoryPropertyFlagBits::eHostVisible |
                     vk::MemoryPropertyFlagBits::eHostCoherent,
                 buffer, bufferMem);
    uniformBuffers.emplace_back(std::move(buffer));
    uniformBuffersMemory.emplace_back(std::move(bufferMem));
    uniformBuffersMapped.emplace_back(
        uniformBuffersMemory[i].mapMemory(0, bufferSize));
  }
}

void vkParticle::createComputeDescriptorSetLayout() {
  std::array layoutBindings{
      vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1,
                                     vk::ShaderStageFlagBits::eCompute,
                                     nullptr),
      vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1,
                                     vk::ShaderStageFlagBits::eCompute,
                                     nullptr),
      vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer, 1,
                                     vk::ShaderStageFlagBits::eCompute,
                                     nullptr)};

  vk::DescriptorSetLayoutCreateInfo layoutInfo{
      .bindingCount = static_cast<uint32_t>(layoutBindings.size()),
      .pBindings = layoutBindings.data()};
  computeDescriptorSetLayout =
      vk::raii::DescriptorSetLayout(device, layoutInfo);
}

void vkParticle::createDescriptorPool() {
  std::array poolSize{vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer,
                                             MaxFramesInFlight),
                      vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer,
                                             MaxFramesInFlight * 2)};

  vk::DescriptorPoolCreateInfo poolInfo{};
  poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
  poolInfo.maxSets = MaxFramesInFlight;
  poolInfo.poolSizeCount = poolSize.size();
  poolInfo.pPoolSizes = poolSize.data();
  descriptorPool = vk::raii::DescriptorPool(device, poolInfo);
}

void vkParticle::createComputeDescriptorSets() {
  std::vector<vk::DescriptorSetLayout> layouts(MaxFramesInFlight,
                                               computeDescriptorSetLayout);
  vk::DescriptorSetAllocateInfo allocInfo{};
  allocInfo.descriptorPool = *descriptorPool;
  allocInfo.descriptorSetCount = MaxFramesInFlight;
  allocInfo.pSetLayouts = layouts.data();
  computeDescriptorSets.clear();
  computeDescriptorSets = device.allocateDescriptorSets(allocInfo);

  for (size_t i = 0; i < MaxFramesInFlight; i++) {
    vk::DescriptorBufferInfo bufferInfo(uniformBuffers[i], 0,
                                        sizeof(UniformBufferObject));

    vk::DescriptorBufferInfo storageBufferInfoLastFrame(
        shaderStorageBuffers[(i - 1) % MaxFramesInFlight], 0,
        sizeof(Particle) * ParticleCount);
    vk::DescriptorBufferInfo storageBufferInfoCurrentFrame(
        shaderStorageBuffers[i], 0, sizeof(Particle) * ParticleCount);
    std::array descriptorWrites{
        vk::WriteDescriptorSet{.dstSet = *computeDescriptorSets[i],
                               .dstBinding = 0,
                               .dstArrayElement = 0,
                               .descriptorCount = 1,
                               .descriptorType =
                                   vk::DescriptorType::eUniformBuffer,
                               .pImageInfo = nullptr,
                               .pBufferInfo = &bufferInfo,
                               .pTexelBufferView = nullptr},
        vk::WriteDescriptorSet{.dstSet = *computeDescriptorSets[i],
                               .dstBinding = 1,
                               .dstArrayElement = 0,
                               .descriptorCount = 1,
                               .descriptorType =
                                   vk::DescriptorType::eStorageBuffer,
                               .pImageInfo = nullptr,
                               .pBufferInfo = &storageBufferInfoLastFrame,
                               .pTexelBufferView = nullptr},
        vk::WriteDescriptorSet{.dstSet = *computeDescriptorSets[i],
                               .dstBinding = 2,
                               .dstArrayElement = 0,
                               .descriptorCount = 1,
                               .descriptorType =
                                   vk::DescriptorType::eStorageBuffer,
                               .pImageInfo = nullptr,
                               .pBufferInfo = &storageBufferInfoCurrentFrame,
                               .pTexelBufferView = nullptr},
    };
    device.updateDescriptorSets(descriptorWrites, {});
  }
}

void vkParticle::createComputePipeline() {
  vk::raii::ShaderModule shaderModule =
      createShaderModule(readFile("slang.spv"), device);

  vk::PipelineShaderStageCreateInfo computeShaderStageInfo{
      .stage = vk::ShaderStageFlagBits::eCompute,
      .module = shaderModule,
      .pName = "compMain"};
  vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
      .setLayoutCount = 1, .pSetLayouts = &*computeDescriptorSetLayout};
  computePipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);
  vk::ComputePipelineCreateInfo pipelineInfo{.stage = computeShaderStageInfo,
                                             .layout = *computePipelineLayout};
  computePipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
}

void vkParticle::createComputeCommandBuffers() {
  computeCommandBuffers.clear();
  vk::CommandBufferAllocateInfo allocInfo{};
  allocInfo.commandPool = *commandPool;
  allocInfo.level = vk::CommandBufferLevel::ePrimary;
  allocInfo.commandBufferCount = MaxFramesInFlight;
  computeCommandBuffers = vk::raii::CommandBuffers(device, allocInfo);
}

void vkParticle::recordComputeCommandBuffer() {
  computeCommandBuffers[currentFrame].reset();
  computeCommandBuffers[currentFrame].begin({});
  computeCommandBuffers[currentFrame].bindPipeline(
      vk::PipelineBindPoint::eCompute, computePipeline);
  computeCommandBuffers[currentFrame].bindDescriptorSets(
      vk::PipelineBindPoint::eCompute, computePipelineLayout, 0,
      {computeDescriptorSets[currentFrame]}, {});
  computeCommandBuffers[currentFrame].dispatch(ParticleCount / 256, 1, 1);
  computeCommandBuffers[currentFrame].end();
}
