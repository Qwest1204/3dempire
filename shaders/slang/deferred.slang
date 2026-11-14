import common;

[[vk::binding(0, 0)]] ConstantBuffer<Camera> camera;
[[vk::binding(0, 1)]] ConstantBuffer<Object> object;
[[vk::binding(0, 2)]] ConstantBuffer<Lighting> lighting;

struct VertexInput { float3 position : POSITION; float3 normal : NORMAL; float2 uv : TEXCOORD0; };
struct GBufferVertex {
    float4 position : SV_Position;
    float3 world_normal : TEXCOORD0;
    float4 color_metallic : TEXCOORD1;
    float roughness : TEXCOORD2;
    float2 current_ndc : TEXCOORD3;
    float2 previous_ndc : TEXCOORD4;
    float3 emissive_occlusion : TEXCOORD5;
};
struct GBufferOutput { float4 albedo : SV_Target0; float4 normal_roughness : SV_Target1; float2 motion : SV_Target2; float4 emissive_occlusion : SV_Target3; };

[shader("vertex")]
GBufferVertex gbuffer_vs(VertexInput input) {
    GBufferVertex output;
    float4 current_clip = mul(camera.view_projection, mul(object.model, float4(input.position, 1.0)));
    float4 previous_clip = mul(camera.previous_view_projection, mul(object.previous_model, float4(input.position, 1.0)));
    output.position = current_clip;
    output.world_normal = normalize(mul(object.model, float4(input.normal, 0.0)).xyz);
    output.color_metallic = object.color_metallic;
    output.roughness = object.roughness;
    output.current_ndc = current_clip.xy / current_clip.w;
    output.previous_ndc = previous_clip.xy / previous_clip.w;
    output.emissive_occlusion = object.emissive_occlusion;
    return output;
}

[shader("fragment")]
GBufferOutput gbuffer_fs(GBufferVertex input) {
    GBufferOutput output;
    output.albedo = float4(input.color_metallic.rgb, input.color_metallic.a);
    output.normal_roughness = float4(normalize(input.world_normal), input.roughness);
    output.motion = (input.current_ndc - input.previous_ndc) * 0.5;
    output.emissive_occlusion = float4(input.emissive_occlusion, 1.0);
    return output;
}

[[vk::binding(0, 0)]] Texture2D<float4> albedo_texture;
[[vk::binding(1, 0)]] Texture2D<float4> normal_roughness_texture;
[[vk::binding(2, 0)]] Texture2D<float2> motion_texture;
[[vk::binding(3, 0)]] Texture2D<float> depth_texture;
[[vk::binding(4, 0)]] SamplerState point_sampler;
[[vk::binding(5, 0)]] Texture2D<float4> sky_texture;
[[vk::binding(6, 0)]] Texture2D<float4> emissive_occlusion_texture;

struct FullscreenVertex { float4 position : SV_Position; float2 uv : TEXCOORD0; nointerpolation uint mode : TEXCOORD1; };
[shader("vertex")]
FullscreenVertex fullscreen_vs(uint vertex : SV_VertexID, uint mode : SV_InstanceID) {
    float2 positions[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
    FullscreenVertex output;
    output.position = float4(positions[vertex], 0.0, 1.0);
    output.uv = fullscreen_uv(positions[vertex]);
    output.mode = mode;
    return output;
}

float3 reconstruct_world_position(float2 uv, float depth) {
    float4 clip = float4(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    float4 world = mul(lighting.inverse_view_projection, clip);
    return world.xyz / world.w;
}
float attenuation(float distance, float range) { float falloff = max(1.0 - distance / range, 0.0); return falloff * falloff / max(distance * distance, 1.0); }

#if ENGINE_DEBUG_VIEW
[shader("fragment")]
float4 composite_fs(FullscreenVertex input) : SV_Target0 { return debug_fs(input); }
#else
[shader("fragment")]
float4 composite_fs(FullscreenVertex input) : SV_Target0 {
    float4 packed_albedo = albedo_texture.Sample(point_sampler, input.uv);
    float3 albedo = packed_albedo.rgb;
    float4 emissive_occlusion = emissive_occlusion_texture.Sample(point_sampler, input.uv);
    float metallic = packed_albedo.a;
    float4 packed_normal = normal_roughness_texture.Sample(point_sampler, input.uv);
    float3 normal = normalize(packed_normal.xyz);
    uint width, height; depth_texture.GetDimensions(width, height);
    float depth = depth_texture.Load(int3(int2(input.position.xy), 0));
    if (depth >= 0.999999) return sky_texture.Sample(point_sampler, input.uv);
    float3 world_position = reconstruct_world_position(input.uv, depth);
    float3 light_direction = normalize(-lighting.directional_direction.xyz);
    float3 directional = albedo * lighting.directional_color_intensity.rgb * max(dot(normal, light_direction), 0.0) * lighting.directional_color_intensity.a;
    float3 view_direction = normalize(lighting.camera_position.xyz - world_position);
    float3 half_vector = normalize(light_direction + view_direction);
    float specular = pow(max(dot(normal, half_vector), 0.0), lerp(8.0, 128.0, 1.0 - packed_normal.w));
    float3 point_to_light = lighting.point_position_range.xyz - world_position;
    float point_distance = length(point_to_light);
    float3 point = albedo * lighting.point_color_intensity.rgb * max(dot(normal, point_to_light / max(point_distance, 0.001)), 0.0) * lighting.point_color_intensity.a * attenuation(point_distance, lighting.point_position_range.w);
    float3 spot_to_light = lighting.spot_position_range.xyz - world_position;
    float spot_distance = length(spot_to_light);
    float3 spot_direction = spot_to_light / max(spot_distance, 0.001);
    float cone = smoothstep(lighting.spot_outer.x, lighting.spot_direction_inner.w, dot(normalize(-spot_direction), normalize(lighting.spot_direction_inner.xyz)));
    float3 spot = albedo * lighting.spot_color_intensity.rgb * max(dot(normal, spot_direction), 0.0) * lighting.spot_color_intensity.a * attenuation(spot_distance, lighting.spot_position_range.w) * cone;
    float3 reflected = reflect(-view_direction, normal);
    float horizon = saturate(reflected.y * 0.5 + 0.5);
    float rotation = lighting.environment_ground.w;
    float2 rotated = float2(reflected.x * cos(rotation) - reflected.z * sin(rotation), reflected.x * sin(rotation) + reflected.z * cos(rotation));
    float environment_variation = 0.85 + 0.15 * cos(rotated.x * 5.0 + rotated.y * 3.0);
    float3 environment = lerp(lighting.environment_ground.rgb, lighting.environment_sky.rgb, horizon) * lighting.environment_sky.a * environment_variation;
    float3 diffuse_ibl = albedo * environment * (1.0 - metallic);
    float3 specular_ibl = environment * lerp(float3(0.04), albedo, metallic) * lerp(1.0, 0.15, packed_normal.w);
    float3 color = ((albedo * lighting.ambient_exposure.x * emissive_occlusion.a) + directional + point + spot + specular * lerp(float3(0.04), albedo, metallic) + diffuse_ibl + specular_ibl + emissive_occlusion.rgb) * lighting.ambient_exposure.y;
    float3 mapped = (color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14);
    return float4(pow(max(mapped, 0.0), 1.0 / 2.2), 1.0);
}
#endif

[shader("fragment")]
float4 debug_fs(FullscreenVertex input) : SV_Target0 {
    float3 albedo = albedo_texture.Sample(point_sampler, input.uv).rgb;
    float4 normal_roughness = normal_roughness_texture.Sample(point_sampler, input.uv);
    float2 motion = motion_texture.Sample(point_sampler, input.uv);
    float depth = depth_texture.Load(int3(int2(input.position.xy), 0));
    switch(input.mode) {
        case 1: return float4(albedo, 1.0);
        case 2: return albedo_texture.Sample(point_sampler, input.uv).aaaa;
        case 3: return float4(normal_roughness.xyz * 0.5 + 0.5, 1.0);
        case 4: return normal_roughness.aaaa;
        case 5: return depth.xxxx;
        case 6: return (1.0 - depth).xxxx;
        case 7: return float4(0.5 + motion.x * 20.0, 0.5 + motion.y * 20.0, length(motion) * 20.0, 1.0);
        default: return float4(albedo, 1.0);
    }
}
