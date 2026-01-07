// Copyright (c) 2025-2026 Ewan Crawford

#include "common.hpp"

void vkParticle::createComputeDescriptorSetLayout() {
  // The compute shader uses 3 descriptor sets:
  // * `ConstantBuffer<UniformBuffer>`
  // * `StructuredBuffer<ParticleSSBO>`
  // * `RWStructuredBuffer<ParticleSSBO>`
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
  // Every frame has 1 unfiorm buffer, and 2 storage buffers
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

  // create descriptor sets for each frame in flight
  for (size_t i = 0; i < SMaxFramesInFlight; i++) {
    // Used for first descriptor to `ConstantBuffer<UniformBuffer>`,
    // so link to the uniform buffer.
    vk::DescriptorBufferInfo bufferInfo(MUniformBuffers[i], 0,
                                        sizeof(UniformBufferObject));

    // Create scratch GPU only memory for last frames details, so we know
    // how to update with the current position based on last position
    constexpr uint32_t ParticleCount = SComputeWorkItems * SComputeWorkGroups;
    vk::DescriptorBufferInfo storageBufferInfoLastFrame(
        MShaderStorageBuffers[(i - 1) % SMaxFramesInFlight], 0,
        sizeof(Particle) * ParticleCount);

    // Create scratch GPU only memory for current frames details
    vk::DescriptorBufferInfo storageBufferInfoCurrentFrame(
        MShaderStorageBuffers[i], 0, sizeof(Particle) * ParticleCount);

    std::array descriptorWrites{
        // Uniform buffer descriptor
        vk::WriteDescriptorSet{.dstSet = *MComputeDescriptorSets[i],
                               .dstBinding = 0,
                               .dstArrayElement = 0,
                               .descriptorCount = 1,
                               .descriptorType =
                                   vk::DescriptorType::eUniformBuffer,
                               .pImageInfo = nullptr,
                               .pBufferInfo = &bufferInfo,
                               .pTexelBufferView = nullptr},

        // Storage buffer descriptor, for last frame
        vk::WriteDescriptorSet{.dstSet = *MComputeDescriptorSets[i],
                               .dstBinding = 1,
                               .dstArrayElement = 0,
                               .descriptorCount = 1,
                               .descriptorType =
                                   vk::DescriptorType::eStorageBuffer,
                               .pImageInfo = nullptr,
                               .pBufferInfo = &storageBufferInfoLastFrame,
                               .pTexelBufferView = nullptr},
        // Storage buffer descriptor, for current frame
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
