uniform float2 noise_offset_uv;
uniform float2 noise_scale_uv;
uniform float4 covariance_parameters1;
uniform float4 covariance_parameters2;

float3 rotateStructTensor(float3 T)
{
    // In image coordinates (+y down), the NE and SE axes are
    // (1, -1)/sqrt(2) and (1, 1)/sqrt(2), respectively.
    // Apply R * T * transpose(R) and return (Txx, Tyy, Txy).
    float trace = T.x + T.y;
    return float3(0.5 * trace + T.z, 0.5 * trace - T.z, 0.5 * (T.y - T.x));
}

void analyzeStructTensor(float3 tensor, float var_noise, out float2 e1, out float2 e2, out float strength, out float coherence)
{
    float txx = tensor.x;
    float tyy = tensor.y;
    float txy = tensor.z;
    float trace = txx + tyy;
    float delta = txx - tyy;
    float gap = length(float2(delta, 2.0 * txy));
    // Clip for numerical inaccuracies. Theoretically >= 0
    float l1 = max(0.0, 0.5 * (trace + gap));
    float l2 = max(0.0, 0.5 * (trace - gap));

    // Pick the form that avoids subtracting nearly equal terms.
    if (gap > 0.0)
    {
        float2 direction = delta >= 0.0 ? float2(gap + delta, 2.0 * tensor.z) : float2(2.0 * txy, gap - delta);
        e1 = normalize(direction);
    }
    else
    {
        e1 = float2(1.0, 0.0);
    }
    e2 = float2(-e1.y, e1.x);

    float s1 = sqrt(l1);
    float s2 = sqrt(l2);

    coherence = (s1 - s2) / (s1 + s2 + 1e-6);
    // Wiener shrinkeage for strenght only
    if (l1 > 0.0)
        l1 *= l1 / (l1 + var_noise);
    strength = sqrt(l1);
}

float3 buildPrecisionMatrix(float strength, float coherence, float2 e1, float2 e2)
{
    float inv_sigma_shrink = covariance_parameters1.x;
    float inv_sigma_stretch = covariance_parameters1.y;
    float inv_sigma_base = covariance_parameters1.w;
    float inv_sigma_blurr = covariance_parameters2.x;

    float gradient_clip = covariance_parameters1.z;
    float gradient_transition = covariance_parameters2.y;

    float t_blurr = clamp(1. - gradient_transition * (strength - gradient_clip), 0., 1.); // TODO Add green std

    float tshrink = min(coherence, 5. * strength);
    float s1 = lerp(inv_sigma_base, inv_sigma_shrink, tshrink);
    float s2 = lerp(inv_sigma_base, inv_sigma_stretch, coherence);

    float k1 = lerp(s1, inv_sigma_blurr, t_blurr);
    float k2 = lerp(s2, inv_sigma_blurr, coherence);


    // Pack k1² * e1 * transpose(e1) + k2² * e2 * transpose(e2) as (xx, yy, xy).
    float3 outer1 = float3(e1.x * e1.x, e1.y * e1.y, e1.x * e1.y);
    float3 outer2 = float3(e2.x * e2.x, e2.y * e2.y, e2.x * e2.y);
    return (k1 * k1) * outer1 + (k2 * k2) * outer2;
}
[shader("compute")]
[numthreads(8, 8, 1)]
void computeCovariance(uint3 tid: SV_DispatchThreadID, Texture2D<float> sourceRaw,
                       Texture2D<float4> noise_lut, SamplerState noiseSampler, RWTexture2D<float4> out)
{
    uint width, height;
    out.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height)
        return;

    int2 outTexel = int2(tid.xy);
    float luma[3][3];
    float meanLuma = 0.0;

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
            luma[y + 1][x + 1] = 0.25 * (rggb.x + rggb.y + rggb.z + rggb.w);
            float weight = (x == 0 ? 2.0 : 1.0) * (y == 0 ? 2.0 : 1.0) / 16.0;
            meanLuma += weight * luma[y + 1][x + 1];
        }
    }

    float tnene = 0.;
    float tsese = 0.;
    float tnese = 0.;
    for (int y = 0; y <= 1; ++y)
    {
        for (int x = 0; x <= 1; ++x)
        {
            float gne = luma[y][x + 1] - luma[y + 1][x];
            float gse = luma[y + 1][x + 1] - luma[y][x];
            tnene += gne * gne;
            tsese += gse * gse;
            tnese += gse * gne;
        }
    }
    float3 structTensor = float3(tnene, tsese, tnese) / 8.; // 4 grads, each scaled by sqrt(2) sample dist, squared
    structTensor = rotateStructTensor(structTensor);        // xx, yy, xy
    float strength, coherence;
    float2 e1, e2;
    float2 noiseUV = (float2(meanLuma, 1.0) - noise_offset_uv) * noise_scale_uv;
    float var_noise = noise_lut.SampleLevel(noiseSampler, noiseUV, 0.0).g;
    analyzeStructTensor(structTensor, var_noise, e1, e2, strength, coherence);
    float3 precisionMatrix = buildPrecisionMatrix(strength, coherence, e1, e2);
    out[outTexel] = float4(precisionMatrix, 0.0);
}
