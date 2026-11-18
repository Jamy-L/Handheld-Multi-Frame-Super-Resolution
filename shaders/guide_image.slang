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

    int2 outTexel = int2(tid.xy);
    float3 avg_rgb = 0.;
    float3 m0 = float3(0.);
    float3 m1 = float3(0.);

    for (int y = -1; y <= 1; ++y)
    {
        for (int x = -1; x <= 1; ++x)
        {
            int2 quad = clamp(outTexel + int2(x, y), int2(0, 0), int2(width - 1, height - 1));
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
            avg_rgb += rgb * wx * wy;
        }
    }
    m0 /= 9.;
    m1 /= 9.;
    float3 var3 = m1 - m0 * m0;
    float var = var3.x + var3.y + var3.z;
    float std = sqrt(var) / 3.;
    avg_rgb = RGBToYCoCg(avg_rgb);
    out[outTexel] = float4(avg_rgb, std);
}
