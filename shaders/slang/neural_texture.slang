struct VertexOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

struct ViewSettings
{
    uint hiddenWidth;
    uint parameterCount;
    uint mode;
    float errorScale;
};

[[vk::binding(0, 0)]] StructuredBuffer<float> modelParameters;
[[vk::binding(1, 0)]] Texture2D<float4> sourceTexture;
[[vk::binding(2, 0)]] SamplerState sourceSampler;
[[vk::binding(3, 0)]] ConstantBuffer<ViewSettings> settings;

[shader("vertex")]
VertexOutput vs_main(uint index : SV_VertexID)
{
    float2 positions[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
    VertexOutput output;
    output.position = float4(positions[index], 0.0, 1.0);
    output.uv = float2(positions[index].x * 0.5 + 0.5, 0.5 - positions[index].y * 0.5);
    return output;
}

float3 evaluate_mlp(float2 uv)
{
    uint w0 = 0;
    uint b0 = settings.hiddenWidth * 2;
    uint w1 = b0 + settings.hiddenWidth;
    uint b1 = w1 + settings.hiddenWidth * 3;
    float output[3] = { modelParameters[b1], modelParameters[b1 + 1], modelParameters[b1 + 2] };
    for (uint unit = 0; unit < settings.hiddenWidth; ++unit)
    {
        float hidden = max(0.0, modelParameters[w0 + unit * 2] * uv.x + modelParameters[w0 + unit * 2 + 1] * uv.y + modelParameters[b0 + unit]);
        output[0] += hidden * modelParameters[w1 + unit];
        output[1] += hidden * modelParameters[w1 + settings.hiddenWidth + unit];
        output[2] += hidden * modelParameters[w1 + settings.hiddenWidth * 2 + unit];
    }
    return saturate(float3(output[0], output[1], output[2]));
}

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target0
{
    float3 source = sourceTexture.Sample(sourceSampler, input.uv).rgb;
    float3 reconstruction = evaluate_mlp(input.uv);
    if (settings.mode == 0) return float4(source, 1.0);
    if (settings.mode == 1) return float4(reconstruction, 1.0);
    if (settings.mode == 2) return float4(abs(source - reconstruction), 1.0);
    if (settings.mode == 3) return float4(saturate(abs(source - reconstruction) * settings.errorScale), 1.0);
    return float4(input.uv.x < 0.5 ? source : reconstruction, 1.0);
}
