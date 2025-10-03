// Copyright (c) 2025 Ewan Crawford

#include "common.hpp"
#include <algorithm>
#include <stdexcept>

void vkParticle::pickPhysicalDevice() {
  std::vector<vk::raii::PhysicalDevice> devices =
      MInstance.enumeratePhysicalDevices();
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
        MRequiredDeviceExtension,
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
    MPhysicalDevice = *devIter;
  } else {
    throw std::runtime_error("failed to find a suitable GPU!");
  }
}

void vkParticle::createLogicalDevice() {
  // find the index of the first queue family that supports graphics
  std::vector<vk::QueueFamilyProperties> queueFamilyProperties =
      MPhysicalDevice.getQueueFamilyProperties();

  // get the first index into queueFamilyProperties which supports both graphics
  // and present
  for (uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size();
       qfpIndex++) {
    if ((queueFamilyProperties[qfpIndex].queueFlags &
         vk::QueueFlagBits::eGraphics) &&
        (queueFamilyProperties[qfpIndex].queueFlags &
         vk::QueueFlagBits::eCompute) &&
        MPhysicalDevice.getSurfaceSupportKHR(qfpIndex, *MSurface)) {
      // found a queue family that supports both graphics and present
      MQueueIndex = qfpIndex;
      break;
    }
  }
  if (MQueueIndex == ~0) {
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
      .queueFamilyIndex = MQueueIndex,
      .queueCount = 1,
      .pQueuePriorities = &queuePriority};
  vk::DeviceCreateInfo deviceCreateInfo{
      .pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &deviceQueueCreateInfo,
      .enabledExtensionCount =
          static_cast<uint32_t>(MRequiredDeviceExtension.size()),
      .ppEnabledExtensionNames = MRequiredDeviceExtension.data()};

  MDevice = vk::raii::Device(MPhysicalDevice, deviceCreateInfo);
  MQueue = vk::raii::Queue(MDevice, MQueueIndex, 0);
}

void vkParticle::createImageViews() {
  assert(MSwapChainImageViews.empty());

  vk::ImageViewCreateInfo imageViewCreateInfo{
      .viewType = vk::ImageViewType::e2D,
      .format = MSwapChainSurfaceFormat.format,
      .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};
  for (auto image : MSwapChainImages) {
    imageViewCreateInfo.image = image;
    MSwapChainImageViews.emplace_back(MDevice, imageViewCreateInfo);
  }
}

void vkParticle::cleanupSwapChain() {
  MSwapChainImageViews.clear();
  MSwapChain = nullptr;
}

void vkParticle::recreateSwapChain() {
  int width = 0, height = 0;
  glfwGetFramebufferSize(MWindow, &width, &height);
  while (width == 0 || height == 0) {
    glfwGetFramebufferSize(MWindow, &width, &height);
    glfwWaitEvents();
  }

  MDevice.waitIdle();

  cleanupSwapChain();
  createSwapChain();
  createImageViews();
}
