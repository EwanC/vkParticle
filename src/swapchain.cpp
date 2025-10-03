// Copyright (c) 2025 Ewan Crawford

#include "common.hpp"
#include <algorithm>
#include <stdexcept>

void vkParticle::createSurface() {
  // GLFW only deals with C API
  VkSurfaceKHR vkSurface;
  if (glfwCreateWindowSurface(*MInstance, MWindow, nullptr, &vkSurface) != 0) {
    throw std::runtime_error("failed to create window surface!");
  }
  MSurface = vk::raii::SurfaceKHR(MInstance, vkSurface);
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
  auto surfaceCapabilities =
      MPhysicalDevice.getSurfaceCapabilitiesKHR(*MSurface);
  MSwapChainExtent = chooseSwapExtent(MWindow, surfaceCapabilities);
  MSwapChainSurfaceFormat =
      chooseSwapSurfaceFormat(MPhysicalDevice.getSurfaceFormatsKHR(*MSurface));
  vk::SwapchainCreateInfoKHR swapChainCreateInfo{
      .surface = *MSurface,
      .minImageCount = chooseSwapMinImageCount(surfaceCapabilities),
      .imageFormat = MSwapChainSurfaceFormat.format,
      .imageColorSpace = MSwapChainSurfaceFormat.colorSpace,
      .imageExtent = MSwapChainExtent,
      .imageArrayLayers = 1,
      .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
      .imageSharingMode = vk::SharingMode::eExclusive,
      .preTransform = surfaceCapabilities.currentTransform,
      .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
      .presentMode = chooseSwapPresentMode(
          MPhysicalDevice.getSurfacePresentModesKHR(*MSurface)),
      .clipped = true};

  MSwapChain = vk::raii::SwapchainKHR(MDevice, swapChainCreateInfo);
  MSwapChainImages = MSwapChain.getImages();
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
