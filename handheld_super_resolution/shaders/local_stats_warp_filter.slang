enum LocalStatsWarpFilter : uint
{
    Bilinear = 0,
    Bezier = 1,
    Biquadratic = 2,
};

float4 warpLocalStats(float2 uv, float2 texelSize, Texture2D<float4> texture,
                      SamplerState linearSampler, LocalStatsWarpFilter filter)
{
    // https://developer.nvidia.com/gpugems/gpugems2/part-iii-high-quality-rendering/chapter-20-fast-third-order-texture-filtering
    float2 position = uv / texelSize - 0.5;
    float2 lowUV, highUV, highWeight;
    switch (filter)
    {
    case LocalStatsWarpFilter.Bezier:
    {
        float2 base = floor(position);
        float2 t = position - base;
        float2 u = 1.0 - t;

        // Pair adjacent cubic B-spline weights on each axis for hardware bilinear sampling.
        float2 w0 = u * u * u / 6.0;
        float2 w1 = (4.0 + t * t * (3.0 * t - 6.0)) / 6.0;
        float2 w2 = (4.0 + u * u * (3.0 * u - 6.0)) / 6.0;
        float2 w3 = t * t * t / 6.0;
        highWeight = w2 + w3;
        lowUV = (base - 0.5 + w1 / (w0 + w1)) * texelSize;
        highUV = (base + 1.5 + w3 / highWeight) * texelSize;
        break;
    }
    case LocalStatsWarpFilter.Biquadratic:
    {
        float2 base = floor(position + 0.5);
        float2 t = position - base;
        float2 leftWeight = 0.5 * (0.5 - t) * (0.5 - t);
        highWeight = 0.5 * (0.5 + t) * (0.5 + t);
        lowUV = (base + 0.5 - leftWeight / (1.0 - highWeight)) * texelSize;
        highUV = (base + 1.5) * texelSize;
        break;
    }
    case LocalStatsWarpFilter.Bilinear:
    default:
        return texture.SampleLevel(linearSampler, uv, 0.0);
    }

    float4 top = lerp(texture.SampleLevel(linearSampler, lowUV, 0.0),
                      texture.SampleLevel(linearSampler, float2(highUV.x, lowUV.y), 0.0), highWeight.x);
    float4 bottom = lerp(texture.SampleLevel(linearSampler, float2(lowUV.x, highUV.y), 0.0),
                         texture.SampleLevel(linearSampler, highUV, 0.0), highWeight.x);
    return lerp(top, bottom, highWeight.y);
}
