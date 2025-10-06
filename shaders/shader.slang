// Copyright (c) 2025 Ewan Crawford

struct VertexShaderInput {
  float2 inPosition; // (x,y) position attribute
  float4 inColor;    // (r,g,b,a) color attribute
};

struct VertexShaderOutput {
  float4 pos : SV_Position; // SV prefix means System Value in HLSL
  float pointSize : SV_PointSize; // Like 'gl_PointSize'
  float3 fragColor : COLOR0; // COLOR0 is diffuse lighting
};

struct FragShaderInput {
  float4 pos : SV_POSITION;
  float3 fragColor : COLOR0;
  float2 pointCoord : SV_PointCoord; // Like 'gl_PointCoord'
};

[shader("vertex")] VertexShaderOutput vertMain(VertexShaderInput input) {
  VertexShaderOutput output;
  output.pointSize = 14.0;
  output.pos = float4(input.inPosition, 1.0, 1.0);
  output.fragColor = input.inColor.rgb;
  return output;
}

[shader("fragment")]
float4 fragMain(FragShaderInput input) : SV_Target {
  float2 coord = input.pointCoord - float2(.5);
  return float4(input.fragColor, .5 - length(coord));
}

// Matches host side particle struct
struct Particle {
  float2 position;
  float2 velocity;
  float4 color;
};

struct UniformBuffer {
  float deltaTime;
};
// Constant buffers are faster than structured buffers, and available to
// more pipeline stages, but smaller (64k-ish).
ConstantBuffer<UniformBuffer> ubo;

struct ParticleSSBO {
  Particle particles;
};

// StructuredBuffer is a read-only buffer which cannot be bound to as vertex
// buffers. See HLSL
StructuredBuffer<ParticleSSBO> particlesIn;
// RWStructuredBuffer can be read and written. See HLSL
RWStructuredBuffer<ParticleSSBO> particlesOut;

// 1D kernel with 256 work-items in each dispatched work-group
[shader("compute")][numthreads(256, 1, 1)]
void compMain(uint3 threadId : SV_DispatchThreadID) {
  uint index = threadId.x;

  // Update position based on previous position and speed
  particlesOut[index].particles.position =
      particlesIn[index].particles.position +
      particlesIn[index].particles.velocity.xy * ubo.deltaTime;
  particlesOut[index].particles.velocity =
      particlesIn[index].particles.velocity;

  // Flip movement at window border
  if ((particlesOut[index].particles.position.x <= -1.0) ||
      (particlesOut[index].particles.position.x >= 1.0)) {
    particlesOut[index].particles.velocity.x =
        -particlesOut[index].particles.velocity.x;
  }
  if ((particlesOut[index].particles.position.y <= -1.0) ||
      (particlesOut[index].particles.position.y >= 1.0)) {
    particlesOut[index].particles.velocity.y =
        -particlesOut[index].particles.velocity.y;
  }
}
