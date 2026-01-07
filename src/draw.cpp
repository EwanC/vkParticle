// Copyright (c) 2025-2026 Ewan Crawford

#include "common.hpp"
#include <stdexcept>

void vkParticle::drawFrame() {
  // The image at `imageIndex` will be available when the fence for the
  // current frame is signalled
  auto [result, imageIndex] = MSwapChain.acquireNextImage(
      UINT64_MAX, nullptr, *MInFlightFences[MCurrentFrame]);
  while (vk::Result::eTimeout ==
         MDevice.waitForFences(*MInFlightFences[MCurrentFrame], vk::True,
                               UINT64_MAX)) {
    ;
  }
  // Reset fence back to unsignalled state after it has been signalled.
  MDevice.resetFences(*MInFlightFences[MCurrentFrame]);

  // Update timeline value for this frame such that the graphics pipeline
  // waits on the compute pipeline to finish, rather than being async.
  uint64_t computeWaitValue = MTimelineValue;
  uint64_t computeSignalValue = ++MTimelineValue;
  uint64_t graphicsWaitValue = computeSignalValue;
  uint64_t graphicsSignalValue = ++MTimelineValue;

  // Update uniform buffer with delta time
  updateUniformBuffer(MCurrentFrame);

  // Submit compute work to device
  {
    // Setup compute command-buffer with commands.
    recordComputeCommandBuffer();

    // Set timeline semaphore values
    vk::TimelineSemaphoreSubmitInfo computeTimelineInfo{
        .waitSemaphoreValueCount = 1,
        .pWaitSemaphoreValues = &computeWaitValue,
        .signalSemaphoreValueCount = 1,
        .pSignalSemaphoreValues = &computeSignalValue};

    // Which stage of the pipeline to wait on for semaphores
    vk::PipelineStageFlags waitStages[] = {
        vk::PipelineStageFlagBits::eComputeShader};
    // Submit command-buffer to queue
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
  // Submit graphics work to device
  {
    // Setup compute command-buffer with commands.
    recordGraphicsCommandBuffer(imageIndex);

    // Submit graphics work, waits for compute to finish.
    vk::TimelineSemaphoreSubmitInfo graphicsTimelineInfo{
        .waitSemaphoreValueCount = 1,
        .pWaitSemaphoreValues = &graphicsWaitValue,
        .signalSemaphoreValueCount = 1,
        .pSignalSemaphoreValues = &graphicsSignalValue};

    // Wait at vertex input stage for compute to finish
    vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eVertexInput;

    // Submit command-buffer to queue
    vk::SubmitInfo graphicsSubmitInfo{
        .pNext = &graphicsTimelineInfo,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*MSemaphore,
        .pWaitDstStageMask = &waitStage,
        .commandBufferCount = 1,
        .pCommandBuffers = &*MGraphicsCommandBuffers[MCurrentFrame],
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &*MSemaphore};
    MQueue.submit(graphicsSubmitInfo, nullptr);

    // Present the image (wait for graphics to finish)
    vk::SemaphoreWaitInfo waitInfo{.semaphoreCount = 1,
                                   .pSemaphores = &*MSemaphore,
                                   .pValues = &graphicsSignalValue};

    // Wait for graphics to complete before presenting rendered frame
    while (vk::Result::eTimeout == MDevice.waitSemaphores(waitInfo, UINT64_MAX))
      ;

    // Before an application can display an image it's format must
    // be transitioned to an appropriate layout.
    vk::PresentInfoKHR presentInfo{.waitSemaphoreCount = 0,
                                   .pWaitSemaphores = nullptr,
                                   .swapchainCount = 1,
                                   .pSwapchains = &*MSwapChain,
                                   .pImageIndices = &imageIndex};
    result = MQueue.presentKHR(presentInfo);
    if (result == vk::Result::eErrorOutOfDateKHR ||
        result == vk::Result::eSuboptimalKHR || MFramebufferResized) {
      MFramebufferResized = false;
      recreateSwapChain();
    } else if (result != vk::Result::eSuccess) {
      throw std::runtime_error("failed to present swap chain image!");
    }
  }

  // Update current frame counter
  MCurrentFrame = (MCurrentFrame + 1) % SMaxFramesInFlight;
}
