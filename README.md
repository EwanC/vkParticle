# vkParticle

Vulkan GPU particle simulation using both graphics and compute.
Based on the [Khronos Vulkan Compute Shader Tutorial](https://docs.vulkan.org/tutorial/latest/11_Compute_Shader.html).

Only tested on Ubuntu 24.04 and requires a C++20 compiler to build along with the Vulkan-SDK. See [Development Environment](https://docs.vulkan.org/tutorial/latest/02_Development_environment.html) for more details on the build setup.

Sample command line for building and running application, the validation layer
is enabled in Debug builds:
```sh
$ mkdir build && cd build
$ cmake ../ -GNinja -DCMAKE_CXX_COMPILER=clang++-19
$ ninja
$ cd build && ./vkParticle
```

Press Escape key to exit, or close window in GUI.

![capture](img/capture.gif)
