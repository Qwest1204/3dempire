struct VertexOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

struct SkyUniform
{
    float4x4 cameraToWorld;
};

[[vk::binding(0, 0)]]
ConstantBuffer<SkyUniform> skyUniform;

[shader("vertex")]
VertexOutput vs_main(uint index : SV_VertexID)
{
    float2 positions[3] = {
        float2(-1.0, -1.0),
        float2(3.0, -1.0),
        float2(-1.0, 3.0),
    };
    VertexOutput output;
    output.position = float4(positions[index], 0.0, 1.0);
    // WGSL render-target UVs use a top-left origin.
    output.uv = float2(positions[index].x * 0.5 + 0.5, 0.5 - positions[index].y * 0.5);
    return output;
}

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target0
{
    float2 clip = input.uv * 2.0 - float2(1.0, 1.0);
    // A unit ray in camera space. Translation is deliberately excluded: sky
    // reacts to camera orientation only, as an infinitely distant environment.
    float3 viewRay = normalize(float3(clip.x, -clip.y, -1.0));
    float3 worldRay = normalize(mul(skyUniform.cameraToWorld, float4(viewRay, 0.0)).xyz);
    float3 zenith = float3(0.015, 0.035, 0.11);
    float3 horizon = float3(0.19, 0.36, 0.58);
    float skyHeight = smoothstep(-0.15, 0.8, worldRay.y);
    float3 sky = lerp(horizon, zenith, skyHeight);
    float3 sunDirection = normalize(float3(0.55, 0.48, -0.68));
    float sun = pow(max(dot(worldRay, sunDirection), 0.0), 800.0);
    return float4(sky + sun * float3(1.0, 0.70, 0.35), 1.0);
}
