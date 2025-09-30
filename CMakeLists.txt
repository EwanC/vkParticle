# Copyright (c) 2025 Ewan Crawford

cmake_minimum_required (VERSION 3.29)
project(VkParticle)

set(CMAKE_CXX_SCAN_FOR_MODULES ON)

# Find glm and glfw3 dependencies
find_package (glfw3 REQUIRED)
find_package (glm REQUIRED)

# set up Vulkan C++ module
find_package (Vulkan REQUIRED)
add_library(VulkanCppModule)
add_library(Vulkan::cppm ALIAS VulkanCppModule)

target_compile_definitions(VulkanCppModule PUBLIC
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
    VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
)
target_include_directories(VulkanCppModule
    PRIVATE
    "${Vulkan_INCLUDE_DIR}"
)
target_link_libraries(VulkanCppModule
    PUBLIC
    Vulkan::Vulkan
)

set_target_properties(VulkanCppModule PROPERTIES CXX_STANDARD 20)
target_sources(VulkanCppModule
    PUBLIC
    FILE_SET cxx_modules TYPE CXX_MODULES
    BASE_DIRS
    "${Vulkan_INCLUDE_DIR}"
    FILES
    "${Vulkan_INCLUDE_DIR}/vulkan/vulkan.cppm"
)

# Find slangc executable from vulkan SDK
find_program(SLANGC_EXECUTABLE slangc HINTS $ENV{VULKAN_SDK}/bin REQUIRED)
# Compile slang shader and add as a dependency to a target
function(add_slang_shader_depedency DEP)
  set(SHADER_SOURCES ${CMAKE_SOURCE_DIR}/shaders/shader.slang)
  set(ENTRY_POINTS -entry vertMain -entry fragMain -entry compMain)

  add_custom_command(
    OUTPUT  ${SHADERS_DIR}/slang.spv
    COMMAND ${SLANGC_EXECUTABLE} ${SHADER_SOURCES} -target spirv -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name ${ENTRY_POINTS} -o slang.spv
    WORKING_DIRECTORY ${SHADERS_DIR}
    DEPENDS ${SHADERS_DIR} ${SHADER_SOURCES}
    COMMENT "Compiling Slang Shaders"
    VERBATIM
   )
   add_custom_target(shader DEPENDS ${SHADERS_DIR}/slang.spv)
   add_dependencies(${DEP} shader)
endfunction()

# Create executable from source files
add_subdirectory(src)
add_executable(vkParticle ${VK_PARTICLE_SOURCES})
set_target_properties(vkParticle PROPERTIES CXX_STANDARD 20)
target_link_libraries(vkParticle Vulkan::cppm glfw)

# Add shader dependencies
add_slang_shader_depedency(vkParticle)
