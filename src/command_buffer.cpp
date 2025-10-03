// Copyright (c) 2025 Ewan Crawford

#include "common.hpp"

void vkParticle::createCommandPool() {
  vk::CommandPoolCreateInfo poolInfo{
      .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
      .queueFamilyIndex = MQueueIndex};
  MCommandPool = vk::raii::CommandPool(MDevice, poolInfo);
}

void vkParticle::createCommandBuffers() {
  MCommandBuffers.clear();
  vk::CommandBufferAllocateInfo allocInfo{
      .commandPool = MCommandPool,
      .level = vk::CommandBufferLevel::ePrimary,
      .commandBufferCount = SMaxFramesInFlight};
  MCommandBuffers = vk::raii::CommandBuffers(MDevice, allocInfo);
}

void vkParticle::createComputeCommandBuffers() {
  MComputeCommandBuffers.clear();
  vk::CommandBufferAllocateInfo allocInfo{};
  allocInfo.commandPool = *MCommandPool;
  allocInfo.level = vk::CommandBufferLevel::ePrimary;
  allocInfo.commandBufferCount = SMaxFramesInFlight;
  MComputeCommandBuffers = vk::raii::CommandBuffers(MDevice, allocInfo);
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
      .image = MSwapChainImages[imageIndex],
      .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                           .baseMipLevel = 0,
                           .levelCount = 1,
                           .baseArrayLayer = 0,
                           .layerCount = 1}};
  vk::DependencyInfo dependency_info = {.dependencyFlags = {},
                                        .imageMemoryBarrierCount = 1,
                                        .pImageMemoryBarriers = &barrier};
  MCommandBuffers[MCurrentFrame].pipelineBarrier2(dependency_info);
}

void vkParticle::recordCommandBuffer(uint32_t imageIndex) {
  MCommandBuffers[MCurrentFrame].reset();
  MCommandBuffers[MCurrentFrame].begin({});
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
      .imageView = MSwapChainImageViews[imageIndex],
      .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
      .loadOp = vk::AttachmentLoadOp::eClear,
      .storeOp = vk::AttachmentStoreOp::eStore,
      .clearValue = clearColor};
  vk::RenderingInfo renderingInfo = {
      .renderArea = {.offset = {0, 0}, .extent = MSwapChainExtent},
      .layerCount = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments = &attachmentInfo};

  MCommandBuffers[MCurrentFrame].beginRendering(renderingInfo);
  MCommandBuffers[MCurrentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics,
                                              *MGraphicsPipeline);
  MCommandBuffers[MCurrentFrame].setViewport(
      0, vk::Viewport(0.0f, 0.0f, static_cast<float>(MSwapChainExtent.width),
                      static_cast<float>(MSwapChainExtent.height), 0.0f, 1.0f));
  MCommandBuffers[MCurrentFrame].setScissor(
      0, vk::Rect2D(vk::Offset2D(0, 0), MSwapChainExtent));
  MCommandBuffers[MCurrentFrame].bindVertexBuffers(
      0, {MShaderStorageBuffers[MCurrentFrame]}, {0});
  MCommandBuffers[MCurrentFrame].draw(SParticleCount, 1, 0, 0);
  MCommandBuffers[MCurrentFrame].endRendering();
  // After rendering, transition the swapchain image to PRESENT_SRC
  transition_image_layout(
      imageIndex, vk::ImageLayout::eColorAttachmentOptimal,
      vk::ImageLayout::ePresentSrcKHR,
      vk::AccessFlagBits2::eColorAttachmentWrite,         // srcAccessMask
      {},                                                 // dstAccessMask
      vk::PipelineStageFlagBits2::eColorAttachmentOutput, // srcStage
      vk::PipelineStageFlagBits2::eBottomOfPipe           // dstStage
  );
  MCommandBuffers[MCurrentFrame].end();
}

void vkParticle::recordComputeCommandBuffer() {
  MComputeCommandBuffers[MCurrentFrame].reset();
  MComputeCommandBuffers[MCurrentFrame].begin({});
  MComputeCommandBuffers[MCurrentFrame].bindPipeline(
      vk::PipelineBindPoint::eCompute, MComputePipeline);
  MComputeCommandBuffers[MCurrentFrame].bindDescriptorSets(
      vk::PipelineBindPoint::eCompute, MComputePipelineLayout, 0,
      {MComputeDescriptorSets[MCurrentFrame]}, {});
  MComputeCommandBuffers[MCurrentFrame].dispatch(SParticleCount / 256, 1, 1);
  MComputeCommandBuffers[MCurrentFrame].end();
}
