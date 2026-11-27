#include "local_stats_warp_filter.slang"

uniform bool noise_correction;
uniform float2 noise_offset_uv;
uniform float2 noise_scale_uv;
uniform float4 rejection_params;
uniform float R_boost;
uniform float unblocker_motion_threshold;

[shader("compute")]
[numthreads(8, 8, 1)]
void computeRejectionCoeff(uint3 tid: SV_DispatchThreadID, Texture2D<float4> sourceLocalStats,
                           Texture2D<float4> localStats, SamplerState linearSampler, Texture2D<float4> flow,
                           SamplerState flowSampler, uniform float2 flow_scale_uv, uniform float2 flow_offset_uv,
                           Texture2D<float4> noise_lut, SamplerState noiseSampler, RWTexture2D<float> out,
                           Texture2D<float> unblocker_texture, SamplerState unblocker_sampler,
                           uniform float2 unblocker_uv_scale,
                           uniform LocalStatsWarpFilter local_stats_warp_filter)
{
    uint width, height;
    out.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height)
        return;

    int2 outTexel = int2(tid.xy);
    float2 texel_size = 1.0 / float2(width, height);
    float2 uv = (outTexel + 0.5) * texel_size;
    float2 uv_flow = (uv - flow_offset_uv) * flow_scale_uv;
    float3 local_flow = flow.SampleLevel(flowSampler, uv_flow, 0.0).xyz; // z is local flow magnitude
    float2 uv_mov = uv + local_flow.xy;

    // refStats is pre-filtered
    float4 refStats = sourceLocalStats.Load(int3(outTexel, 0));
    float4 movStats = warpLocalStats(uv_mov, texel_size, localStats, linearSampler, local_stats_warp_filter);

    float std = min(refStats.w, movStats.w); // Min std between both frames
    float3 color_distances = refStats.xyz - movStats.xyz;
    color_distances *= color_distances;
    float boost = 1.0;
    if (noise_correction)
    {
        float2 noiseUV = (float2(refStats.x, 0.0) - noise_offset_uv) * noise_scale_uv;
        float4 noise = noise_lut.SampleLevel(noiseSampler, noiseUV, 0.0);
        std = max(std, noise.w); // noise-induced std
        // Two independent guides contribute noise to their squared difference.
        color_distances *= color_distances / max(color_distances + 2.0 * noise.xyz, 1e-30); // Wiener shrinkeage
        // The 2.* is bcs the noiseParam is the avg diff between one noisy patch and the gt clean. So diff between 2
        // noisy is the double
        // If large local flow motion, and local std > k * noise std: apply bigger R_boost
        if (local_flow.z >= rejection_params.z && std >= rejection_params.w * noise.w)
            boost = R_boost;
    }
    float color_distance = color_distances.x + color_distances.y + color_distances.z;

    float R_mult = rejection_params.x;
    float R_threshold = rejection_params.y;
    float r = color_distance / max(std * std, 1e-30);
    float weight = exp2(min(-r * R_mult * boost + R_threshold, 0.));

    float unblocker = 0.0;
    if (local_flow.z > unblocker_motion_threshold)
        unblocker = unblocker_texture.SampleLevel(unblocker_sampler, uv * unblocker_uv_scale, 0.0);
    out[outTexel] = min(1.0 - unblocker, weight);
}
