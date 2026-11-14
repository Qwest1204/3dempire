// Neural-texture experiment: 14 -> 32 -> 32 -> RGB.
// Input  = x, y, sin/cos(2x, 4x, 8x, 2y, 4y, 8y).
// Output = exp(clamp(linear, -16, 0)) — mirrors CPU training exactly.
// Weights are supplied at runtime; nothing is embedded.

// ---- Layout constants -------------------------------------------------------
static const uint kInputFeatures  = 14;
static const uint kHiddenUnits    = 32;
static const uint kOutputChannels = 3;

// Parameter buffer layout (floats), in order:
//   W0 [32][14]  B0 [32]
//   W1 [32][32]  B1 [32]
//   W2 [ 3][32]  B2 [ 3]
static const uint kOffW0 = 0;
static const uint kOffB0 = kOffW0 + kHiddenUnits * kInputFeatures;
static const uint kOffW1 = kOffB0 + kHiddenUnits;
static const uint kOffB1 = kOffW1 + kHiddenUnits * kHiddenUnits;
static const uint kOffW2 = kOffB1 + kHiddenUnits;
static const uint kOffB2 = kOffW2 + kOutputChannels * kHiddenUnits;

// ---- Render modes -----------------------------------------------------------
static const uint kModeSource        = 0;
static const uint kModeReconstruction = 1;
static const uint kModeAbsError      = 2;
static const uint kModeScaledError   = 3;
// anything else -> split-screen

// ---- Resources --------------------------------------------------------------
struct ViewSettings {
    uint  parameterCount;
    uint  mode;
    float errorScale;
    float _padding;
};

struct VertexOutput {
    float4 position : SV_Position;
    float2 uv       : TEXCOORD0;
};

[[vk::binding(0, 0)]] StructuredBuffer<float>  parameters;
[[vk::binding(1, 0)]] Texture2D<float4>        sourceTexture;
[[vk::binding(2, 0)]] SamplerState             sourceSampler;
[[vk::binding(3, 0)]] ConstantBuffer<ViewSettings> settings;

// ---- Vertex stage -----------------------------------------------------------
[shader("vertex")]
VertexOutput vs_main(uint index : SV_VertexID) {
    // Fullscreen triangle (covers clip space without a vertex buffer).
    float2 positions[3] = {
        float2(-1.0, -1.0),
        float2( 3.0, -1.0),
        float2(-1.0,  3.0)
    };
    float2 p = positions[index];

    VertexOutput output;
    output.position = float4(p, 0.0, 1.0);
    output.uv       = float2(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
    return output;
}

// ---- Neural network ---------------------------------------------------------
// Dense layer: out[unit] = act( b[unit] + sum_f W[unit][f] * in[f] )
// `W` is row-major: W[unit][feature] at W[unit * inFeatures + feature].
float layer_forward_unit(uint unit,
                         uint weightBase, uint biasBase,
                         uint inFeatures,
                         const float inVec[/*inFeatures*/])
{
    float value = parameters[biasBase + unit];
    for (uint f = 0; f < inFeatures; ++f) {
        value += parameters[weightBase + unit * inFeatures + f] * inVec[f];
    }
    return max(0.0, value); // ReLU
}

float3 evaluate_fourier_mlp(float2 uv) {
    // 14-D Fourier input vector.
    float input[kInputFeatures] = {
        uv.x, uv.y,
        sin(2.0 * uv.x), cos(2.0 * uv.x),
        sin(2.0 * uv.y), cos(2.0 * uv.y),
        sin(4.0 * uv.x), cos(4.0 * uv.x),
        sin(4.0 * uv.y), cos(4.0 * uv.y),
        sin(8.0 * uv.x), cos(8.0 * uv.x),
        sin(8.0 * uv.y), cos(8.0 * uv.y)
    };

    float h0[kHiddenUnits];
    float h1[kHiddenUnits];

    for (uint u = 0; u < kHiddenUnits; ++u)
        h0[u] = layer_forward_unit(u, kOffW0, kOffB0, kInputFeatures, input);

    for (uint u = 0; u < kHiddenUnits; ++u)
        h1[u] = layer_forward_unit(u, kOffW1, kOffB1, kHiddenUnits, h0);

    float3 output = float3(parameters[kOffB2], parameters[kOffB2 + 1], parameters[kOffB2 + 2]);
    for (uint u = 0; u < kHiddenUnits; ++u) {
        float h = h1[u];
        output.x += parameters[kOffW2 + 0 * kHiddenUnits + u] * h;
        output.y += parameters[kOffW2 + 1 * kHiddenUnits + u] * h;
        output.z += parameters[kOffW2 + 2 * kHiddenUnits + u] * h;
    }

    // Exponential final activation; clamp keeps exp() in a safe range.
    return exp(clamp(output, -16.0, 0.0));
}

// ---- Color-space helpers ----------------------------------------------------
float3 srgb_to_linear(float3 v) {
    return select(v / 12.92, pow((v + 0.055) / 1.055, 2.4), v > 0.04045);
}

float3 linear_to_srgb(float3 v) {
    return select(v * 12.92, 1.055 * pow(max(v, 0.0), 1.0 / 2.4) - 0.055, v > 0.0031308);
}

// ---- Fragment stage ---------------------------------------------------------
[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target0 {
    float3 sourceDisplay   = sourceTexture.Sample(sourceSampler, input.uv).rgb;
    float3 sourceLinear    = srgb_to_linear(sourceDisplay);
    float3 reconstruction  = evaluate_fourier_mlp(input.uv);
    float3 absError        = abs(sourceLinear - reconstruction);

    switch (settings.mode) {
        case kModeSource:
            return float4(sourceDisplay, 1.0);

        case kModeReconstruction:
            return float4(linear_to_srgb(reconstruction), 1.0);

        case kModeAbsError:
            return float4(linear_to_srgb(absError), 1.0);

        case kModeScaledError:
            return float4(linear_to_srgb(saturate(absError * settings.errorScale)), 1.0);

        default: // split-screen: left = source, right = reconstruction
            return float4(input.uv.x < 0.5 ? sourceDisplay
                                          : linear_to_srgb(reconstruction),
                          1.0);
    }
}