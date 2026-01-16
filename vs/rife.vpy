import os

import vapoursynth as vs
from vapoursynth import core
from vsmlrt import RIFE, Backend, RIFEModel

# Options
MAX_SHORT_EDGE = 1440
MULTI = 2
PADDING_MULTI = 32
MODEL = RIFEModel.v4_6
SC_MODE = 1
SOURCE_FPS_LIMIT = 31
BACKEND = Backend.TRT(
    fp16=True,
    force_fp16=True,
    tf32=True,
    workspace=None,
    num_streams=2,
    use_cuda_graph=True,
    output_format=1,
    static_shape=False,
    # custom_env={"LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", "")},
)

clip = video_in

if container_fps > SOURCE_FPS_LIMIT:
    raise Exception(
        f"Source FPS ({container_fps}) exceeds the limit ({SOURCE_FPS_LIMIT})"
    )

# Scene Change Detection: Make sure the input is not a duplicated frame
if SC_MODE == 0:
    clip = clip
elif SC_MODE == 1:
    clip = core.misc.SCDetect(clip=clip, threshold=0.15)
elif SC_MODE == 2:
    sup = core.mv.Super(clip=clip, pel=1)
    vec = core.mv.Analyse(super=sup, isb=True)
    clip = core.mv.SCDetection(clip=clip, vectors=vec, thscd1=240, thscd2=130)

# Downscale if the short edge exceeds the limit
if clip.width >= clip.height:
    short_edge = clip.height
    long_edge = clip.width * (MAX_SHORT_EDGE / clip.height)
    w_ds = long_edge
    h_ds = MAX_SHORT_EDGE
else:
    short_edge = clip.width
    long_edge = clip.height * (MAX_SHORT_EDGE / clip.width)
    w_ds = MAX_SHORT_EDGE
    h_ds = long_edge

# Resize and convert to vs.RGBH
if short_edge > MAX_SHORT_EDGE:
    clip = clip.resize.Bicubic(
        width=w_ds, height=h_ds, format=vs.RGBH, matrix_in_s="709", range_in=1
    )
else:
    clip = clip.resize.Bilinear(format=vs.RGBH, matrix_in_s="709", range_in=1)


# Pad to multiple of PADDING_MULTI
pad_w = (PADDING_MULTI - (clip.width % PADDING_MULTI)) % PADDING_MULTI
pad_h = (PADDING_MULTI - (clip.height % PADDING_MULTI)) % PADDING_MULTI
if pad_w > 0 or pad_h > 0:
    clip = core.std.AddBorders(clip, right=pad_w, bottom=pad_h)

# RIFE
clip = RIFE(
    clip=clip,
    multi=MULTI,
    model=MODEL,
    backend=BACKEND,
)

# Remove padding
clip = core.std.Crop(clip, right=pad_w, bottom=pad_h)

# Convert back to vs.YUV420P8
clip = clip.resize.Bilinear(format=vs.YUV420P8, matrix_s="709", range=1)

clip.set_output()
