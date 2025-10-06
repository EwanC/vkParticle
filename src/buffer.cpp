// Copyright (c) 2025 Ewan Crawford

#include "common.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <glm/gtc/matrix_transform.hpp>
#include <random>
#include <stdexcept>

void vkParticle::updateUniformBuffer(uint32_t currentImage) {
  // Update uniform buffer with a new time delta.
  UniformBufferObject ubo{};
  // `MLastFrameTime` set on each iteration of vkParticle::mainLoop()
  ubo.deltaTime = static_cast<float>(MLastFrameTime) * 2.f;
  memcpy(MUniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
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

  // Create a single-submit command-buffer containing a single copy command
  // for the full size of the src/dst buffers
  vk::CommandBufferAllocateInfo allocInfo{.commandPool = MCommandPool,
                                          .level =
                                              vk::CommandBufferLevel::ePrimary,
                                          .commandBufferCount = 1};
  vk::raii::CommandBuffer commandCopyBuffer =
      std::move(MDevice.allocateCommandBuffers(allocInfo).front());
  commandCopyBuffer.begin(vk::CommandBufferBeginInfo{
      .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  commandCopyBuffer.copyBuffer(*srcBuffer, *dstBuffer,
                               vk::BufferCopy(0, 0, size));
  commandCopyBuffer.end();
  MQueue.submit(vk::SubmitInfo{.commandBufferCount = 1,
                               .pCommandBuffers = &*commandCopyBuffer},
                nullptr);
  MQueue.waitIdle();
}

void vkParticle::createShaderStorageBuffers() {
  // Setup random distribution to use for particle initial locations
  std::default_random_engine rndEngine(static_cast<unsigned>(time(nullptr)));
  std::uniform_real_distribution rndDist(0.0f, 1.0f);

  // Initialize host memory with particle instances
  std::vector<Particle> particles(SParticleCount);
  for (auto &particle : particles) {
    // Initial particle positions on a circle
    float r = 0.25f * sqrtf(rndDist(rndEngine));
    float theta = rndDist(rndEngine) * 2.0f * 3.14159265358979323846f;
    float x = r * cosf(theta) * SWindowHeight / SWindowWidth;
    float y = r * sinf(theta);
    particle.position = glm::vec2(x, y);
    particle.velocity = normalize(glm::vec2(x, y)) * 0.00025f;
    particle.color = glm::vec4(rndDist(rndEngine), rndDist(rndEngine),
                               rndDist(rndEngine), 1.0f);
  }

  // Memory required for a buffer of all particles
  vk::DeviceSize bufferSize = sizeof(Particle) * SParticleCount;

  // Create a host-visible staging buffer used to upload data to the gpu
  vk::raii::Buffer stagingBuffer({});
  vk::raii::DeviceMemory stagingBufferMemory({});
  createBuffer(MDevice, MPhysicalDevice, bufferSize,
               vk::BufferUsageFlagBits::eTransferSrc,
               vk::MemoryPropertyFlagBits::eHostVisible |
                   vk::MemoryPropertyFlagBits::eHostCoherent,
               stagingBuffer, stagingBufferMemory);

  // Map staging buffer buffer, and copy host std::vector data into it.
  void *dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
  memcpy(dataStaging, particles.data(), (size_t)bufferSize);
  stagingBufferMemory.unmapMemory();

  MShaderStorageBuffers.clear();
  MShaderStorageBuffersMemory.clear();

  // Use single-shot command-buffer to copy initial particle data from
  // temporary buffers to shader storage buffers.
  // SSBs have usage flag bits set for all of storage, vertex, and transfer,
  // so that they can be used in vertex shader and compute shader, and
  // data transferred from host to GPU (for UBO with delta time).
  for (size_t i = 0; i < SMaxFramesInFlight; i++) {
    vk::raii::Buffer shaderStorageBufferTemp({});
    vk::raii::DeviceMemory shaderStorageBufferTempMemory({});
    createBuffer(MDevice, MPhysicalDevice, bufferSize,
                 vk::BufferUsageFlagBits::eStorageBuffer |
                     vk::BufferUsageFlagBits::eVertexBuffer |
                     vk::BufferUsageFlagBits::eTransferDst,
                 vk::MemoryPropertyFlagBits::eDeviceLocal, // GPU resident
                 shaderStorageBufferTemp, shaderStorageBufferTempMemory);
    copyBuffer(stagingBuffer, shaderStorageBufferTemp, bufferSize);
    MShaderStorageBuffers.emplace_back(std::move(shaderStorageBufferTemp));
    MShaderStorageBuffersMemory.emplace_back(
        std::move(shaderStorageBufferTempMemory));
  }
}

void vkParticle::createUniformBuffers() {
  MUniformBuffers.clear();
  MUniformBuffersMemory.clear();
  MUniformBuffersMapped.clear();

  // Each frame has a host visible/coherent uniform buffer
  // that is persistently mapped. This is used to pass in the
  // new time value to the compute shader, rather than passing
  // this through the vertex buffer and updating that every frame.
  for (size_t i = 0; i < SMaxFramesInFlight; i++) {
    vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
    vk::raii::Buffer buffer({});
    vk::raii::DeviceMemory bufferMem({});
    createBuffer(MDevice, MPhysicalDevice, bufferSize,
                 vk::BufferUsageFlagBits::eUniformBuffer,
                 vk::MemoryPropertyFlagBits::eHostVisible |
                     vk::MemoryPropertyFlagBits::eHostCoherent,
                 buffer, bufferMem);
    MUniformBuffers.emplace_back(std::move(buffer));
    MUniformBuffersMemory.emplace_back(std::move(bufferMem));
    MUniformBuffersMapped.emplace_back(
        MUniformBuffersMemory[i].mapMemory(0, bufferSize));
  }
}
