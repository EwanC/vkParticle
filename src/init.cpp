// Copyright (c) 2025 Ewan Crawford

#include "common.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>

const std::vector<const char *> vkParticle::SValidationLayers = {
    "VK_LAYER_KHRONOS_validation"};

namespace {
void framebufferResizeCallback(GLFWwindow *window, int width, int height) {
  auto app = reinterpret_cast<vkParticle *>(glfwGetWindowUserPointer(window));
  app->MFramebufferResized = true;
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

  MWindow = glfwCreateWindow(SWindowWidth, SWindowHeight, "vkParticle", nullptr,
                             nullptr);
  glfwSetWindowUserPointer(MWindow, this);
  glfwSetFramebufferSizeCallback(MWindow, framebufferResizeCallback);
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
  createGraphicsCommandBuffers();
  createComputeCommandBuffers();
  createSyncObjects();
}

void vkParticle::cleanup() {
  glfwDestroyWindow(MWindow);
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
  if (SEnableValidationLayers) {
    requiredLayers.assign(SValidationLayers.begin(), SValidationLayers.end());
  }

  auto layerProperties = MContext.enumerateInstanceLayerProperties();
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
  auto requiredExtensions = getRequiredExtensions(SEnableValidationLayers);
  auto extensionProperties = MContext.enumerateInstanceExtensionProperties();
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
  MInstance = vk::raii::Instance(MContext, createInfo);
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
  if (!SEnableValidationLayers)
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

  MDebugMessenger =
      MInstance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
}

void vkParticle::mainLoop() {
  // Exit on escape key press or GUI window close
  while (glfwGetKey(MWindow, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
         !glfwWindowShouldClose(MWindow)) {
    glfwPollEvents();
    drawFrame();
    // We want to animate the particle system using the last frames time to get
    // smooth, frame-rate independent animation
    double currentTime = glfwGetTime();
    MLastFrameTime = (currentTime - MLastTime) * 1000.0;
    MLastTime = currentTime;
  }
  MDevice.waitIdle();
}

void vkParticle::createSyncObjects() {
  MInFlightFences.clear();

  vk::SemaphoreTypeCreateInfo semaphoreType{
      .semaphoreType = vk::SemaphoreType::eTimeline, .initialValue = 0};
  MSemaphore = vk::raii::Semaphore(MDevice, {.pNext = &semaphoreType});
  MTimelineValue = 0;

  for (size_t i = 0; i < SMaxFramesInFlight; i++) {
    vk::FenceCreateInfo fenceInfo{};
    MInFlightFences.emplace_back(MDevice, fenceInfo);
  }
}
