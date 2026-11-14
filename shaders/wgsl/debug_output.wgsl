struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0)
    );
    var output: VertexOutput;
    output.position = vec4(positions[index], 0.0, 1.0);
    output.uv = positions[index] * 0.5 + vec2(0.5);
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let horizon = mix(vec3(0.015, 0.025, 0.06), vec3(0.08, 0.22, 0.38), input.uv.y);
    let grid = smoothstep(0.985, 1.0, abs(sin(input.uv.x * 85.0))) + smoothstep(0.985, 1.0, abs(sin(input.uv.y * 55.0)));
    return vec4(horizon + grid * vec3(0.03, 0.08, 0.13), 1.0);
}
