// Copyright (c) 2025 Ewan Crawford

#include "common.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <glm/gtc/matrix_transform.hpp>
#include <random>
#include <stdexcept>

void vkParticle::updateUniformBuffer(uint32_t currentImage) {
  UniformBufferObject ubo{};
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
  // Initialize particles
  std::default_random_engine rndEngine(static_cast<unsigned>(time(nullptr)));
  std::uniform_real_distribution rndDist(0.0f, 1.0f);

  // Initial particle positions on a circle
  std::vector<Particle> particles(SParticleCount);
  for (auto &particle : particles) {
    float r = 0.25f * sqrtf(rndDist(rndEngine));
    float theta = rndDist(rndEngine) * 2.0f * 3.14159265358979323846f;
    float x = r * cosf(theta) * SWindowHeight / SWindowWidth;
    float y = r * sinf(theta);
    particle.position = glm::vec2(x, y);
    particle.velocity = normalize(glm::vec2(x, y)) * 0.00025f;
    particle.color = glm::vec4(rndDist(rndEngine), rndDist(rndEngine),
                               rndDist(rndEngine), 1.0f);
  }

  vk::DeviceSize bufferSize = sizeof(Particle) * SParticleCount;

  // Create a staging buffer used to upload data to the gpu
  vk::raii::Buffer stagingBuffer({});
  vk::raii::DeviceMemory stagingBufferMemory({});
  createBuffer(MDevice, MPhysicalDevice, bufferSize,
               vk::BufferUsageFlagBits::eTransferSrc,
               vk::MemoryPropertyFlagBits::eHostVisible |
                   vk::MemoryPropertyFlagBits::eHostCoherent,
               stagingBuffer, stagingBufferMemory);

  void *dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
  memcpy(dataStaging, particles.data(), (size_t)bufferSize);
  stagingBufferMemory.unmapMemory();

  MShaderStorageBuffers.clear();
  MShaderStorageBuffersMemory.clear();

  // Copy initial particle data to all storage buffers
  for (size_t i = 0; i < SMaxFramesInFlight; i++) {
    vk::raii::Buffer shaderStorageBufferTemp({});
    vk::raii::DeviceMemory shaderStorageBufferTempMemory({});
    createBuffer(MDevice, MPhysicalDevice, bufferSize,
                 vk::BufferUsageFlagBits::eStorageBuffer |
                     vk::BufferUsageFlagBits::eVertexBuffer |
                     vk::BufferUsageFlagBits::eTransferDst,
                 vk::MemoryPropertyFlagBits::eDeviceLocal,
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
