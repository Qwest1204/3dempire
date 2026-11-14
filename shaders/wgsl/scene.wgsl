struct Camera { view_projection: mat4x4<f32> };
struct Object { model: mat4x4<f32>, color: vec4<f32> };

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<uniform> object: Object;

struct VertexInput { @location(0) position: vec3<f32>, @location(1) normal: vec3<f32> };
struct VertexOutput { @builtin(position) position: vec4<f32>, @location(0) normal: vec3<f32>, @location(1) color: vec3<f32> };

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = camera.view_projection * object.model * vec4(input.position, 1.0);
    output.normal = normalize((object.model * vec4(input.normal, 0.0)).xyz);
    output.color = object.color.rgb;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let light = normalize(vec3<f32>(0.4, 0.8, 0.3));
    let diffuse = max(dot(normalize(input.normal), light), 0.0);
    let color = input.color * (0.14 + diffuse * 0.86);
    return vec4(pow(color, vec3(1.0 / 2.2)), 1.0);
}
