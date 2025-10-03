// Copyright (c) 2025 Ewan Crawford

#include "common.hpp"

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

void vkParticle::createComputeCommandBuffers() {
  computeCommandBuffers.clear();
  vk::CommandBufferAllocateInfo allocInfo{};
  allocInfo.commandPool = *commandPool;
  allocInfo.level = vk::CommandBufferLevel::ePrimary;
  allocInfo.commandBufferCount = MaxFramesInFlight;
  computeCommandBuffers = vk::raii::CommandBuffers(device, allocInfo);
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
