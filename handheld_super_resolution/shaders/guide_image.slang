#include "local_stats_warp_filter.slang"

uniform float k_green_diff_impact;

float3 RGBToYCoCg(float3 rgb)
{
    // Green contains the sum of the two Bayer green samples.
    float y = 0.25 * (rgb.r + rgb.g + rgb.b);
    float cg = 0.25 * (rgb.g - rgb.r - rgb.b);
    float co = 0.5 * (rgb.r - rgb.b);
    return float3(y, cg, co);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void computeGuide(uint3 tid: SV_DispatchThreadID, Texture2D<float> sourceRaw, RWTexture2D<float4> out)
{
    uint width, height;
    out.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height)
        return;

    int2 maxTexel = int2(width - 1, height - 1);
    int2 outTexel = int2(tid.xy);
    float3 avg_rgb = 0.;
    float3 m0 = float3(0.);
    float3 m1 = float3(0.);
    float green_differences = 0.0;

    for (int y = -1; y <= 1; ++y)
    {
        for (int x = -1; x <= 1; ++x)
        {
            int2 quad = clamp(outTexel + int2(x, y), int2(0, 0), maxTexel);
            float r = sourceRaw.Load(int3(2 * quad.x, 2 * quad.y, 0));
            float g1 = sourceRaw.Load(int3(2 * quad.x + 1, 2 * quad.y, 0));
            float g2 = sourceRaw.Load(int3(2 * quad.x, 2 * quad.y + 1, 0));
            float b = sourceRaw.Load(int3(2 * quad.x + 1, 2 * quad.y + 1, 0));

            float4 rggb = float4(r, g1, g2, b);
            rggb = sqrt(max(0., rggb));
            float3 rgb = float3(rggb.x, rggb.y + rggb.z, rggb.w);
            m0 += rgb;
            m1 += rgb * rgb;
            float wx = x == 0. ? .5 : .25;
            float wy = y == 0. ? .5 : .25;
            float w = wx * wy;
            avg_rgb += rgb * w;
            green_differences += abs(rggb.y-rggb.z) * w;
        }
    }
    m0 /= 9.;
    m1 /= 9.;
    float3 var3 = m1 - m0 * m0;
    float var = var3.x + var3.y + var3.z;
    float std = sqrt(var) / 3.;

    std = max(std, green_differences);
    avg_rgb = RGBToYCoCg(avg_rgb);

    // Blend the YCoCg with a green only variant, depending on how strong the green gradients are.
    float3 greens = float3(0.5 * avg_rgb.y, 0., 0.);
    float green_diff_impact = min(1.0, k_green_diff_impact * green_differences);
    float3 reference_color = lerp(avg_rgb, greens, green_diff_impact);

    out[outTexel] = float4(reference_color, std);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void blurLocalStats(uint3 tid: SV_DispatchThreadID, Texture2D<float4> stats, RWTexture2D<float4> out,
                    SamplerState linearSampler, uniform LocalStatsWarpFilter local_stats_warp_filter)
{
    uint width, height;
    out.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height)
        return;

    float2 texelSize = 1.0 / float2(width, height);
    float2 uv = (float2(tid.xy) + 0.5) * texelSize;
    out[tid.xy] = warpLocalStats(uv, texelSize, stats, linearSampler, local_stats_warp_filter);
}
