// Copyright (c) 2025 Ewan Crawford

#include "common.hpp"
#include <stdexcept>

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
