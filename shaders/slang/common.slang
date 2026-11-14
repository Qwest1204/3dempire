// Shared ABI definitions for all production Slang shaders. Rust uniform types
// intentionally use the same field order and std140-compatible alignment.
struct Camera
{
    float4x4 view_projection;
    float4x4 previous_view_projection;
};

struct Object
{
    float4x4 model;
    float4x4 previous_model;
    float4 color_metallic;
    float roughness;
    float3 emissive_occlusion;
};

struct Lighting
{
    float4x4 inverse_view_projection;
    float4 camera_position;
    float4 directional_direction;
    float4 directional_color_intensity;
    float4 point_position_range;
    float4 point_color_intensity;
    float4 spot_position_range;
    float4 spot_direction_inner;
    float4 spot_color_intensity;
    float4 spot_outer;
    float4 ambient_exposure;
    float4 environment_sky;
    float4 environment_ground;
};

float2 fullscreen_uv(float2 position)
{
    return float2(position.x * 0.5 + 0.5, 0.5 - position.y * 0.5);
}
