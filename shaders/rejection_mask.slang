uniform bool noise_correction;
uniform float2 noise_offset_uv;
uniform float2 noise_scale_uv;
uniform float4 rejection_params;
uniform float R_boost;

[shader("compute")]
[numthreads(8, 8, 1)]
void computeRejectionCoeff(uint3 tid: SV_DispatchThreadID, Texture2D<float4> sourceLocalStats,
                           Texture2D<float4> localStats, SamplerState statsSampler, Texture2D<float4> flow,
                           SamplerState flowSampler, uniform float2 flow_scale_uv, uniform float2 flow_offset_uv,
                           Texture2D<float4> noise_lut, SamplerState noiseSampler, RWTexture2D<float> out)
{
    uint width, height;
    out.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height)
        return;

    int2 outTexel = int2(tid.xy);
    float2 uv = (outTexel + 0.5) / float2(width, height);
    float2 uv_flow = (uv - flow_offset_uv) * flow_scale_uv;
    float3 local_flow = flow.SampleLevel(flowSampler, uv_flow, 0.0).xyz; // z is local flow magnitude
    float2 uv_mov = uv + local_flow.xy;

    // TODO:here implement a proper sampling (bicubic or biquadratic) using hardware bilinear.
    // Apply the same filter to the warped image and ref
    float4 refStats = sourceLocalStats.SampleLevel(statsSampler, uv, 0.0);
    float4 movStats = localStats.SampleLevel(statsSampler, uv_mov, 0.0);

    float std = min(refStats.w, movStats.w);
    float3 color_distances = refStats.xyz - movStats.xyz;
    color_distances *= color_distances;
    float boost = 1.0;
    if (noise_correction)
    {
        float2 noiseUV = (float2(refStats.x, 0.0) - noise_offset_uv) * noise_scale_uv;
        float4 noise = noise_lut.SampleLevel(noiseSampler, noiseUV, 0.0);
        std = max(std, noise.w);
        // Two independent guides contribute noise to their squared difference.
        color_distances *= color_distances / max(color_distances + 2.0 * noise.xyz, 1e-30);
        if (local_flow.z >= rejection_params.z && std >= rejection_params.w * noise.w)
            boost = R_boost;
    }
    float color_distance = color_distances.x + color_distances.y + color_distances.z;

    float R_mult = rejection_params.x;
    float R_threshold = rejection_params.y;
    float r = color_distance / max(std * std, 1e-30);
    float weight = exp2(min(-r * R_mult * boost + R_threshold, 0.));

    out[outTexel] = weight;
}
