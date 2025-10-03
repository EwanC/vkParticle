// Copyright (c) 2025 Ewan Crawford

#include "common.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>

const std::vector<const char *> vkParticle::validationLayers = {
    "VK_LAYER_KHRONOS_validation"};

namespace {
void framebufferResizeCallback(GLFWwindow *window, int width, int height) {
  auto app = reinterpret_cast<vkParticle *>(glfwGetWindowUserPointer(window));
  app->framebufferResized = true;
}
} // anonymous namespace

void vkParticle::run() {
  initWindow();
  initVulkan();
  mainLoop();
  cleanup();
}

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

void vkParticle::cleanup() {
  glfwDestroyWindow(window);
  glfwTerminate();
}

namespace {
std::vector<const char *> getRequiredExtensions(bool enableValidationLayers) {
  uint32_t glfwExtCount = 0;
  auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtCount);

  std::vector extensions(glfwExtensions, glfwExtensions + glfwExtCount);
  if (enableValidationLayers) {
    extensions.push_back(vk::EXTDebugUtilsExtensionName);
  }
  return extensions;
}
} // namespace

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
  auto requiredExtensions = getRequiredExtensions(enableValidationLayers);
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
