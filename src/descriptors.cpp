// Copyright (c) 2025 Ewan Crawford

#include "common.hpp"

void vkParticle::createComputeDescriptorSetLayout() {
  std::array layoutBindings{
      vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1,
                                     vk::ShaderStageFlagBits::eCompute,
                                     nullptr),
      vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1,
                                     vk::ShaderStageFlagBits::eCompute,
                                     nullptr),
      vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer, 1,
                                     vk::ShaderStageFlagBits::eCompute,
                                     nullptr)};

  vk::DescriptorSetLayoutCreateInfo layoutInfo{
      .bindingCount = static_cast<uint32_t>(layoutBindings.size()),
      .pBindings = layoutBindings.data()};
  computeDescriptorSetLayout =
      vk::raii::DescriptorSetLayout(device, layoutInfo);
}

void vkParticle::createDescriptorPool() {
  std::array poolSize{vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer,
                                             MaxFramesInFlight),
                      vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer,
                                             MaxFramesInFlight * 2)};

  vk::DescriptorPoolCreateInfo poolInfo{};
  poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
  poolInfo.maxSets = MaxFramesInFlight;
  poolInfo.poolSizeCount = poolSize.size();
  poolInfo.pPoolSizes = poolSize.data();
  descriptorPool = vk::raii::DescriptorPool(device, poolInfo);
}

void vkParticle::createComputeDescriptorSets() {
  std::vector<vk::DescriptorSetLayout> layouts(MaxFramesInFlight,
                                               computeDescriptorSetLayout);
  vk::DescriptorSetAllocateInfo allocInfo{};
  allocInfo.descriptorPool = *descriptorPool;
  allocInfo.descriptorSetCount = MaxFramesInFlight;
  allocInfo.pSetLayouts = layouts.data();
  computeDescriptorSets.clear();
  computeDescriptorSets = device.allocateDescriptorSets(allocInfo);

  for (size_t i = 0; i < MaxFramesInFlight; i++) {
    vk::DescriptorBufferInfo bufferInfo(uniformBuffers[i], 0,
                                        sizeof(UniformBufferObject));

    vk::DescriptorBufferInfo storageBufferInfoLastFrame(
        shaderStorageBuffers[(i - 1) % MaxFramesInFlight], 0,
        sizeof(Particle) * ParticleCount);
    vk::DescriptorBufferInfo storageBufferInfoCurrentFrame(
        shaderStorageBuffers[i], 0, sizeof(Particle) * ParticleCount);
    std::array descriptorWrites{
        vk::WriteDescriptorSet{.dstSet = *computeDescriptorSets[i],
                               .dstBinding = 0,
                               .dstArrayElement = 0,
                               .descriptorCount = 1,
                               .descriptorType =
                                   vk::DescriptorType::eUniformBuffer,
                               .pImageInfo = nullptr,
                               .pBufferInfo = &bufferInfo,
                               .pTexelBufferView = nullptr},
        vk::WriteDescriptorSet{.dstSet = *computeDescriptorSets[i],
                               .dstBinding = 1,
                               .dstArrayElement = 0,
                               .descriptorCount = 1,
                               .descriptorType =
                                   vk::DescriptorType::eStorageBuffer,
                               .pImageInfo = nullptr,
                               .pBufferInfo = &storageBufferInfoLastFrame,
                               .pTexelBufferView = nullptr},
        vk::WriteDescriptorSet{.dstSet = *computeDescriptorSets[i],
                               .dstBinding = 2,
                               .dstArrayElement = 0,
                               .descriptorCount = 1,
                               .descriptorType =
                                   vk::DescriptorType::eStorageBuffer,
                               .pImageInfo = nullptr,
                               .pBufferInfo = &storageBufferInfoCurrentFrame,
                               .pTexelBufferView = nullptr},
    };
    device.updateDescriptorSets(descriptorWrites, {});
  }
}
