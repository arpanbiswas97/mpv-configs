import math

import vapoursynth as vs
from vapoursynth import core

# Options
MAX_SHORT_EDGE = 1080
SOURCE_FPS_LIMIT = 32
MODEL = 23
FACTOR_NUM = 2
FACTOR_DEN = 1
GPU_ID = 0
GPU_THREADS = 2
SC_MODE = 1
SKIP = True
SKIP_THRESHOLD = 60.0

clip = video_in
colorlv = clip.get_frame(0).props._ColorRange
fmt_src = clip.format

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

# Make sure the dimensions are even if the input is subsampled
if fmt_src.subsampling_w or fmt_src.subsampling_h:
    if not (w_ds % 2 == 0):
        w_ds = math.floor(w_ds / 2) * 2
    if not (h_ds % 2 == 0):
        h_ds = math.floor(h_ds / 2) * 2

# Resize and convert to vs.RGBS
if short_edge > MAX_SHORT_EDGE:
    clip = clip.resize.Bicubic(
        width=w_ds, height=h_ds, format=vs.RGBS, matrix_in_s="709"
    )
else:
    clip = clip.resize.Bilinear(format=vs.RGBS, matrix_in_s="709")

# RIFE
clip = core.rife.RIFE(
    clip=clip,
    model=MODEL,
    factor_num=FACTOR_NUM,
    factor_den=FACTOR_DEN,
    gpu_id=GPU_ID,
    gpu_thread=GPU_THREADS,
    sc=True if SC_MODE else False,
    skip=SKIP,
    skip_threshold=SKIP_THRESHOLD,
)

# Convert back to vs.YUV420P8
clip = clip.resize.Bilinear(
    format=fmt_src.id, matrix_s="709", range=1 if colorlv == 0 else None
)

clip.set_output()
