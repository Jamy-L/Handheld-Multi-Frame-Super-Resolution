float neighborhoodMinimum(int2 pixel, int2 lastPixel, Texture2D<float> source)
{
    float minimum = source.Load(int3(pixel, 0));
    for (int y = -2; y <= 2; ++y)
    {
        for (int x = -2; x <= 2; ++x)
        {
            int2 tap = clamp(pixel + int2(x, y), int2(0, 0), lastPixel);
            minimum = min(minimum, source.Load(int3(tap, 0)));
        }
    }
    return minimum;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void min5x5(uint3 tid : SV_DispatchThreadID, Texture2D<float> source, RWTexture2D<float> out)
{
    uint width, height;
    out.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height)
        return;

    int2 pixel = int2(tid.xy);
    out[pixel] = neighborhoodMinimum(pixel, int2(width, height) - 1, source);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void min5x5AndAccumulate(uint3 tid : SV_DispatchThreadID, Texture2D<float> source,
                        RWTexture2D<float> out, RWTexture2D<float> accumulated_rejection)
{
    uint width, height;
    out.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height)
        return;

    int2 pixel = int2(tid.xy);
    float weight = neighborhoodMinimum(pixel, int2(width, height) - 1, source);
    out[pixel] = weight;
    accumulated_rejection[pixel] += weight;
}
