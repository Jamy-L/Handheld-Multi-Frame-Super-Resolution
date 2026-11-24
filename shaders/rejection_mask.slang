uniform float2 noise_offset_uv;
uniform float2 noise_scale_uv;
uniform float4 rejection_params;

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

    // TODO  here implement a proper sampling (bicubic or biquadratic) using hardware bilinear.
    // Apply the same filter to the warped image and ref
    float4 refStats = sourceLocalStats.SampleLevel(statsSampler, uv, 0.0);
    float4 movStats = localStats.SampleLevel(statsSampler, uv_mov, 0.0);

    float2 noiseUV = (float2(refStats.x, 0.0) - noise_offset_uv) * noise_scale_uv;
    float4 noiseParams = noise_lut.SampleLevel(noiseSampler, noiseUV, 0.0);

    float std = min(refStats.w, movStats.w); // Min std between both frames
    float noise_std = noiseParams.w;
    std = max(std, noise_std);           // noise-induced std

    float3 color_distances = (refStats.xyz - movStats.xyz);
    color_distances *= color_distances;
    color_distances *= color_distances / (color_distances + 2. * noiseParams.xyz); // Wiener shrinkeage
    // The 2.* is bcs the noiseParam is the avg diff between one noisy patch and the gt clean. So diff between 2 noisy
    // is the double
    float color_distance = color_distances.x + color_distances.y + color_distances.z;

    float R_mult = rejection_params.x;
    float R_threshold = rejection_params.y;
    float M_th = rejection_params.z;
    float k_std = rejection_params.w;
    float local_flow_variation = local_flow.z;
    // If large local flow motion, and local std > k * noise std: apply bigger R_boost
    bool big_motion = false;
    if (local_flow_variation >= M_th)
    {
        big_motion = true;
    }

    float R_boost = 1.;
    if (big_motion && std >= k_std * noise_std) {
        R_boost = 6.;
      }

    float r = color_distance / (std * std);
    float weight = exp2(min(-r * R_mult * R_boost + R_threshold, 0.));

    out[outTexel] = weight;
}
