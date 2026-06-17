#!/usr/bin/env python3
"""
视频操作指示器(HUD)叠加工具 — WASD按键 + 方向遥感
支持自定义按键时序，适用于 WorldPlay 和 Matrix-Game 两类视频。
"""

import math, subprocess, os, sys
import cv2
import imageio_ffmpeg
import numpy as np

# ═══════════════════════════════════════════
#  用户可调参数
# ═══════════════════════════════════════════

# 超采样倍率(2=高质量, 1=快速)
HUD_SIZE = 2

# ── 颜色 ──
KEY_HL        = (245, 245, 245)   # 高亮文字（白灰）
KEY_DIM       = (60, 60, 60)      # 未激活文字（深灰）
KEY_BG_HL     = (160, 160, 160)   # 高亮背景
KEY_BG_DIM    = (25, 25, 25)      # 未激活背景
KEY_BORDER    = (90, 90, 90)      # 未激活边框
KEY_BORDER_HL = (200, 200, 200)   # 高亮边框

JOYSTICK_BG   = (18, 18, 18)      # 遥感背景
JOYSTICK_LINE = (100, 100, 100)   # 遥感线条
JOYSTICK_BRIGHT = (230, 230, 230) # 遥感高亮
KNOB_COLOR    = (200, 200, 200)   # 遥感旋钮
KNOB_INNER    = (40, 40, 40)      # 旋钮内圈

# ── 透明度 ──
ALPHA_ACTIVE   = 0.85    # 按键激活
ALPHA_INACTIVE = 0.12    # 按键未激活
TRANSITION     = 0.15    # 遥感平滑过渡时间(秒)

# ── 布局偏移 ──
# Matrix 视频(1280x704)需右移20px，WorldPlay(848x480)不移
X_SHIFT = 0  # 默认偏移量，处理 Matrix 时传 20

# ═══════════════════════════════════════════
#  时序辅助函数
# ═══════════════════════════════════════════

def make_state_fn(segments):
    """创建时序函数。

    参数:
        segments: [(按键名, 开始秒, 结束秒), ...]
        按键名: "W"/"A"/"S"/"D" = WASD高亮, "left"/"right" = 遥感方向

    返回:
        函数 fn(t) → (active_key, joystick_dir, show_left, show_right)
    """
    def fn(t):
        for k, ts, te in segments:
            if ts <= t < te:
                if k in ("W","S","A","D"):
                    return (k, 0.0, False, False)
                d = t - ts
                if d < TRANSITION:
                    v = -_ss(d/TRANSITION) if k == "left" else _ss(d/TRANSITION)
                else:
                    v = -1.0 if k == "left" else 1.0
                return (None, v, k=="left", k=="right")
        return (None, 0.0, False, False)
    return fn


# ═══════════════════════════════════════════
#  内置预设时序
# ═══════════════════════════════════════════

# WorldPlay — 海滩(前后): W(0-5s)→S(5-10.54s)
BEACH_FN = make_state_fn([("W", 0, 5), ("S", 5, 10.54)])

# WorldPlay — 湖(左右): 遥感 left(0-5s)→right(5-10.54s)
LAKE_FN = make_state_fn([("left", 0, 5), ("right", 5, 10.54)])

# Matrix-Game — 城堡: W(0-10.41s)→S(10.41-19.82s)
CASTLE_FN = make_state_fn([("W", 0, 10.41), ("S", 10.41, 19.82)])

# Matrix-Game — 雪地: 遥感 left(0-5.7s)→right(5.7-10.41s)
SNOW_FN = make_state_fn([("left", 0, 5.7), ("right", 5.7, 10.41)])

# 首页视频(127潜在帧): W→遥感左→W→遥感右→W
HERO_FN = make_state_fn([
    ("W", 0, 31/6), ("left", 31/6, 47/6),
    ("W", 47/6, 63/6), ("right", 63/6, 95/6),
    ("W", 95/6, 127/6)
])

# 通用: A(0-5.7s)→D(5.7-结束)
A_D_FN = make_state_fn([("A", 0, 5.7), ("D", 5.7, 10.41)])

# 通用: 遥感 right(0-5.2s)→left(5.2-结束)
RIGHT_LEFT_FN = make_state_fn([("right", 0, 5.2), ("left", 5.2, 10.54)])

# 通用: 遥感 left(0-5.2s)→right(5.2-结束)
LEFT_RIGHT_FN = make_state_fn([("left", 0, 5), ("right", 5, 10.54)])


# ═══════════════════════════════════════════
#  核心绘制函数
# ═══════════════════════════════════════════

def _ss(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)

def _rounded_rect(img, x, y, w, h, r, color, alpha):
    ov = img.copy()
    cv2.rectangle(ov, (x+r, y), (x+w-r, y+h), color, -1)
    cv2.rectangle(ov, (x, y+r), (x+w, y+h-r), color, -1)
    for cx, cy in [(x+r, y+r), (x+w-r, y+r), (x+r, y+h-r), (x+w-r, y+h-r)]:
        cv2.circle(ov, (cx, cy), r, color, -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)

def _rounded_rect_outline(img, x, y, w, h, r, color, t):
    cv2.line(img, (x+r, y), (x+w-r, y), color, t, cv2.LINE_AA)
    cv2.line(img, (x+r, y+h), (x+w-r, y+h), color, t, cv2.LINE_AA)
    cv2.line(img, (x, y+r), (x, y+h-r), color, t, cv2.LINE_AA)
    cv2.line(img, (x+w, y+r), (x+w, y+h-r), color, t, cv2.LINE_AA)
    cv2.ellipse(img, (x+r, y+r), (r, r), 180, 0, 90, color, t, cv2.LINE_AA)
    cv2.ellipse(img, (x+w-r, y+r), (r, r), 270, 0, 90, color, t, cv2.LINE_AA)
    cv2.ellipse(img, (x+w-r, y+h-r), (r, r), 0, 0, 90, color, t, cv2.LINE_AA)
    cv2.ellipse(img, (x+r, y+h-r), (r, r), 90, 0, 90, color, t, cv2.LINE_AA)

def _draw_key(img, cx, cy, size, label, active):
    """绘制单个按键"""
    h = size // 2
    x, y = cx - h, cy - h
    bg = KEY_BG_HL if active else KEY_BG_DIM
    fg = KEY_HL if active else KEY_DIM
    border = KEY_BORDER_HL if active else KEY_BORDER
    alpha = ALPHA_ACTIVE if active else ALPHA_INACTIVE
    radius = max(2, size // 7)
    _rounded_rect(img, x, y, size, size, radius, bg, alpha)
    _rounded_rect_outline(img, x, y, size, size, radius, border, max(1, size//50))
    font_scale = max(0.3, size / 55)
    thick = max(1, size // 20)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
    tx = x + (size - tw) // 2
    ty = y + (size + th) // 2 + 2
    cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, fg, thick, cv2.LINE_AA)

def _draw_arc(img, center, r, sd, ed, color, t):
    """绘制弧线"""
    if ed < sd:
        ed += 360
    pts = []
    for d in np.linspace(sd, ed, 36):
        a = math.radians(d % 360)
        pts.append((int(center[0] + math.cos(a)*r), int(center[1] + math.sin(a)*r)))
    if len(pts) > 1:
        cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, color, t, cv2.LINE_AA)

def _draw_joystick(img, cx, cy, direction, show_left, show_right):
    """绘制遥感罗盘"""
    h, w = img.shape[:2]
    s = w / 848.0
    r = int(55 * s)
    knob_r = max(3, int(5 * s))
    axis_len = int(r * 0.50)
    offset = direction * r * 0.25
    knob = (int(cx + offset), cy)

    # 背景
    ov = img.copy()
    cv2.circle(ov, (cx, cy), r, JOYSTICK_BG, -1)
    cv2.addWeighted(ov, 0.25, img, 0.75, 0, img)

    # 外圈 + 十字线
    cv2.circle(img, (cx, cy), r, JOYSTICK_LINE, 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), int(r*0.85), JOYSTICK_LINE, 1, cv2.LINE_AA)
    cv2.line(img, (cx-axis_len, cy), (cx+axis_len, cy), JOYSTICK_LINE, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy-axis_len), (cx, cy+axis_len), JOYSTICK_LINE, 1, cv2.LINE_AA)

    # 方向箭头
    arrow = max(4, int(7 * s))
    lc = JOYSTICK_BRIGHT if show_left else JOYSTICK_LINE
    rc = JOYSTICK_BRIGHT if show_right else JOYSTICK_LINE
    up_c = down_c = JOYSTICK_LINE
    cv2.fillConvexPoly(img, np.array([(cx, cy-axis_len-arrow), (cx-arrow, cy-axis_len+4),
                                      (cx+arrow, cy-axis_len+4)], dtype=np.int32), up_c, cv2.LINE_AA)
    cv2.fillConvexPoly(img, np.array([(cx, cy+axis_len+arrow), (cx-arrow, cy+axis_len-4),
                                      (cx+arrow, cy+axis_len-4)], dtype=np.int32), down_c, cv2.LINE_AA)
    cv2.fillConvexPoly(img, np.array([(cx-axis_len-arrow, cy), (cx-axis_len+4, cy-arrow),
                                      (cx-axis_len+4, cy+arrow)], dtype=np.int32), lc, cv2.LINE_AA)
    cv2.fillConvexPoly(img, np.array([(cx+axis_len+arrow, cy), (cx+axis_len-4, cy-arrow),
                                      (cx+axis_len-4, cy+arrow)], dtype=np.int32), rc, cv2.LINE_AA)

    # 高亮弧
    if show_left:
        _draw_arc(img, (cx, cy), int(r*0.92), 132, 226, JOYSTICK_BRIGHT, max(1, int(2*s)))
    elif show_right:
        _draw_arc(img, (cx, cy), int(r*0.92), -46, 48, JOYSTICK_BRIGHT, max(1, int(2*s)))

    # 旋钮
    gl = img.copy()
    cv2.circle(gl, knob, int(knob_r*1.5), JOYSTICK_BRIGHT, -1, cv2.LINE_AA)
    cv2.addWeighted(gl, 0.10, img, 0.90, 0, img)
    cv2.circle(img, knob, knob_r, KNOB_COLOR, max(1, int(2*s)), cv2.LINE_AA)
    cv2.circle(img, knob, int(knob_r*0.4), KNOB_INNER, -1, cv2.LINE_AA)

def draw_hud(img, t, state_fn, x_shift=0):
    """在主帧上绘制完整的 HUD（WASD + 遥感）。

    参数:
        img:      OpenCV BGR 图像帧
        t:         当前时间(秒)
        state_fn:  时序函数(由 make_state_fn 创建)
        x_shift:   水平偏移(Matrix视频需20, WorldPlay用0)
    """
    h, w = img.shape[:2]
    s = w / 848.0
    x0 = int((35 + x_shift) * s)
    y0 = h - int(95 * s)
    step = int(58 * s)
    ks = int(50 * s)

    ak, jd, sl, sr = state_fn(t)

    # WASD 按键（左下方）
    _draw_key(img, x0 + step, y0, ks, "W", ak == "W")
    _draw_key(img, x0, y0 + step, ks, "A", ak == "A")
    _draw_key(img, x0 + step, y0 + step, ks, "S", ak == "S")
    _draw_key(img, x0 + 2 * step, y0 + step, ks, "D", ak == "D")

    # 遥感（WASD 右侧）
    r = int(55 * s)
    wasd_bottom = y0 + step + ks
    wasd_right = x0 + 2 * step + ks // 2
    joy_cx = wasd_right + int(25 * s) + r
    joy_cy = wasd_bottom - r - int(25 * s)
    _draw_joystick(img, joy_cx, joy_cy, jd, sl, sr)


# ═══════════════════════════════════════════
#  视频处理
# ═══════════════════════════════════════════

class H264Writer:
    """FFmpeg 原始帧写入器"""
    def __init__(self, path, width, height, fps):
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "warning",
               "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{width}x{height}", "-r", str(fps),
               "-i", "-", "-an", "-c:v", "libx264",
               "-preset", "medium", "-crf", "23",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame):
        self.proc.stdin.write(frame.tobytes())

    def close(self):
        self.proc.stdin.close()
        code = self.proc.wait()
        if code:
            raise RuntimeError(f"ffmpeg exit code {code}")


def process_video(input_path, output_path, state_fn, x_shift=0):
    """处理单个视频，添加 HUD。

    参数:
        input_path:  输入视频路径
        output_path: 输出视频路径
        state_fn:    时序函数
        x_shift:     水平偏移(Matrix=20, WorldPlay=0)
    """
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  {os.path.basename(output_path)}: {w}x{h}, {fps}fps, {total}帧")

    writer = H264Writer(output_path, w, h, fps)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if HUD_SIZE > 1:
            hi = cv2.resize(frame, (w*HUD_SIZE, h*HUD_SIZE),
                           interpolation=cv2.INTER_CUBIC)
            draw_hud(hi, idx / fps, state_fn, x_shift)
            frame = cv2.resize(hi, (w, h), interpolation=cv2.INTER_AREA)
        else:
            draw_hud(frame, idx / fps, state_fn, x_shift)
        writer.write(frame)
        idx += 1
        if idx % int(fps * 2) == 0:
            print(f"    {idx}/{total} ({idx*100//total}%)", flush=True)

    writer.close()
    cap.release()
    print(f"  ✓ {os.path.basename(output_path)}")
    return True


# ═══════════════════════════════════════════
#  main — 命令行入口
# ═══════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="为视频添加 WASD+遥感 HUD")
    parser.add_argument("input", help="输入视频路径")
    parser.add_argument("output", help="输出视频路径")
    parser.add_argument("--preset", choices=["beach","lake","castle","snow","hero","ad","rl","lr"],
                        help="使用内置时序预设")
    parser.add_argument("--type", choices=["worldplay","matrix"], default="worldplay",
                        help="视频类型(影响HUD位置偏移)")
    parser.add_argument("--shift", type=int, default=None,
                        help="手动指定水平偏移像素(覆盖--type)")
    args = parser.parse_args()

    # 选择时序
    preset_map = {
        "beach": BEACH_FN, "lake": LAKE_FN,
        "castle": CASTLE_FN, "snow": SNOW_FN,
        "hero": HERO_FN, "ad": A_D_FN,
        "rl": RIGHT_LEFT_FN, "lr": LEFT_RIGHT_FN,
    }
    state_fn = preset_map.get(args.preset)
    if state_fn is None:
        print("请使用 --preset 指定时序预设")
        sys.exit(1)

    # 偏移
    if args.shift is not None:
        x_shift = args.shift
    else:
        x_shift = 20 if args.type == "matrix" else 0

    process_video(args.input, args.output, state_fn, x_shift)
    print("完成")
