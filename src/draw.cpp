// Copyright (c) 2025 Ewan Crawford

#include "common.hpp"
#include <stdexcept>

void vkParticle::drawFrame() {
  auto [result, imageIndex] = MSwapChain.acquireNextImage(
      UINT64_MAX, nullptr, *MInFlightFences[MCurrentFrame]);
  while (vk::Result::eTimeout ==
         MDevice.waitForFences(*MInFlightFences[MCurrentFrame], vk::True,
                               UINT64_MAX)) {
    ;
  }
  MDevice.resetFences(*MInFlightFences[MCurrentFrame]);

  // Update timeline value for this frame
  uint64_t computeWaitValue = MTimelineValue;
  uint64_t computeSignalValue = ++MTimelineValue;
  uint64_t graphicsWaitValue = computeSignalValue;
  uint64_t graphicsSignalValue = ++MTimelineValue;

  updateUniformBuffer(MCurrentFrame);

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

    vk::SubmitInfo computeSubmitInfo{
        .pNext = &computeTimelineInfo,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*MSemaphore,
        .pWaitDstStageMask = waitStages,
        .commandBufferCount = 1,
        .pCommandBuffers = &*MComputeCommandBuffers[MCurrentFrame],
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &*MSemaphore};

    MQueue.submit(computeSubmitInfo, nullptr);
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
                                      .pWaitSemaphores = &*MSemaphore,
                                      .pWaitDstStageMask = &waitStage,
                                      .commandBufferCount = 1,
                                      .pCommandBuffers =
                                          &*MCommandBuffers[MCurrentFrame],
                                      .signalSemaphoreCount = 1,
                                      .pSignalSemaphores = &*MSemaphore};

    MQueue.submit(graphicsSubmitInfo, nullptr);

    // Present the image (wait for graphics to finish)
    vk::SemaphoreWaitInfo waitInfo{.semaphoreCount = 1,
                                   .pSemaphores = &*MSemaphore,
                                   .pValues = &graphicsSignalValue};

    // Wait for graphics to complete before presenting
    while (vk::Result::eTimeout == MDevice.waitSemaphores(waitInfo, UINT64_MAX))
      ;

    vk::PresentInfoKHR presentInfo{.waitSemaphoreCount =
                                       0, // No binary semaphores needed
                                   .pWaitSemaphores = nullptr,
                                   .swapchainCount = 1,
                                   .pSwapchains = &*MSwapChain,
                                   .pImageIndices = &imageIndex};

    result = MQueue.presentKHR(presentInfo);
    if (result == vk::Result::eErrorOutOfDateKHR ||
        result == vk::Result::eSuboptimalKHR || framebufferResized) {
      framebufferResized = false;
      recreateSwapChain();
    } else if (result != vk::Result::eSuccess) {
      throw std::runtime_error("failed to present swap chain image!");
    }
  }

  MCurrentFrame = (MCurrentFrame + 1) % SMaxFramesInFlight;
}
