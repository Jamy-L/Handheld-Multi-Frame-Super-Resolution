[shader("compute")]
[numthreads(8, 8, 1)]
void fusion(uint3 tid: SV_DispatchThreadID, Texture2D<float> sourceRaw, Texture2D<float4> alignments,
            SamplerState flowSampler, Texture2D<float4> precision, SamplerState precisionSampler,
            Texture2D<float> robustness, SamplerState robustnessSampler, RWTexture2D<float4> numerator,
            RWTexture2D<float4> denominator, uniform bool bayerMode, uniform float2 flowOffsetUV,
            uniform float2 flowScaleUV)
{
    uint width, height;
    numerator.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height)
        return;
    uint lr_width, lr_height;
    sourceRaw.GetDimensions(lr_width, lr_height);

    int2 outputPixel = int2(tid.xy);
    float2 uv = (float2(tid.xy) + 0.5) / float2(width, height);
    float2 uv_flow = (uv - flowOffsetUV) * flowScaleUV;
    float2 flow = alignments.SampleLevel(flowSampler, uv_flow, 0.0).xy;
    float r = robustness.SampleLevel(robustnessSampler, uv, 0.0);

    float2 uv_mov = uv + flow; // TODO make sure the flow is passed in uv units
    float3 precisionMatrix = precision.SampleLevel(precisionSampler, uv_mov, 0.0).xyz;
    float2x2 sigmaInv = float2x2(precisionMatrix.x, precisionMatrix.z,
                                precisionMatrix.z, precisionMatrix.y);

    float2 ji_mov = uv_mov * int2(lr_width, lr_height) - 0.5;
    int2 ji_center = int2(round(ji_mov));
    float4 numDelta = 0.0;
    float4 denDelta = 0.0;
    for (int di = -1; di <= 1; ++di)
    {
        for (int dj = -1; dj <= 1; ++dj)
        {
            int2 ji = ji_center + int2(dj, di);
            if (any(ji < 0) || any(ji >= int2(lr_width, lr_height)))
                continue;

            // RGGB parity sums to R=0, G=1, B=2.
            int channel = bayerMode ? (ji.x & 1) + (ji.y & 1) : 0;
            float2 d = float2(ji) - ji_mov;
            float distanceSquared = max(0.0, dot(d, mul(sigmaInv, d)));
            float weight = r ;//* exp(-0.5 * distanceSquared);
            numDelta[channel] += weight * sourceRaw.Load(int3(ji, 0));
            denDelta[channel] += weight;
        }
    }
    numerator[outputPixel] = numerator[outputPixel] + numDelta;
    denominator[outputPixel] = denominator[outputPixel] + denDelta;
}
