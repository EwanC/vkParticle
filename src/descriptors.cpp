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
  MComputeDescriptorSetLayout =
      vk::raii::DescriptorSetLayout(MDevice, layoutInfo);
}

void vkParticle::createDescriptorPool() {
  std::array poolSize{vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer,
                                             SMaxFramesInFlight),
                      vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer,
                                             SMaxFramesInFlight * 2)};

  vk::DescriptorPoolCreateInfo poolInfo{};
  poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
  poolInfo.maxSets = SMaxFramesInFlight;
  poolInfo.poolSizeCount = poolSize.size();
  poolInfo.pPoolSizes = poolSize.data();
  MDescriptorPool = vk::raii::DescriptorPool(MDevice, poolInfo);
}

void vkParticle::createComputeDescriptorSets() {
  std::vector<vk::DescriptorSetLayout> layouts(SMaxFramesInFlight,
                                               MComputeDescriptorSetLayout);
  vk::DescriptorSetAllocateInfo allocInfo{};
  allocInfo.descriptorPool = *MDescriptorPool;
  allocInfo.descriptorSetCount = SMaxFramesInFlight;
  allocInfo.pSetLayouts = layouts.data();
  MComputeDescriptorSets.clear();
  MComputeDescriptorSets = MDevice.allocateDescriptorSets(allocInfo);

  for (size_t i = 0; i < SMaxFramesInFlight; i++) {
    vk::DescriptorBufferInfo bufferInfo(MUniformBuffers[i], 0,
                                        sizeof(UniformBufferObject));

    vk::DescriptorBufferInfo storageBufferInfoLastFrame(
        MShaderStorageBuffers[(i - 1) % SMaxFramesInFlight], 0,
        sizeof(Particle) * SParticleCount);
    vk::DescriptorBufferInfo storageBufferInfoCurrentFrame(
        MShaderStorageBuffers[i], 0, sizeof(Particle) * SParticleCount);
    std::array descriptorWrites{
        vk::WriteDescriptorSet{.dstSet = *MComputeDescriptorSets[i],
                               .dstBinding = 0,
                               .dstArrayElement = 0,
                               .descriptorCount = 1,
                               .descriptorType =
                                   vk::DescriptorType::eUniformBuffer,
                               .pImageInfo = nullptr,
                               .pBufferInfo = &bufferInfo,
                               .pTexelBufferView = nullptr},
        vk::WriteDescriptorSet{.dstSet = *MComputeDescriptorSets[i],
                               .dstBinding = 1,
                               .dstArrayElement = 0,
                               .descriptorCount = 1,
                               .descriptorType =
                                   vk::DescriptorType::eStorageBuffer,
                               .pImageInfo = nullptr,
                               .pBufferInfo = &storageBufferInfoLastFrame,
                               .pTexelBufferView = nullptr},
        vk::WriteDescriptorSet{.dstSet = *MComputeDescriptorSets[i],
                               .dstBinding = 2,
                               .dstArrayElement = 0,
                               .descriptorCount = 1,
                               .descriptorType =
                                   vk::DescriptorType::eStorageBuffer,
                               .pImageInfo = nullptr,
                               .pBufferInfo = &storageBufferInfoCurrentFrame,
                               .pTexelBufferView = nullptr},
    };
    MDevice.updateDescriptorSets(descriptorWrites, {});
  }
}
