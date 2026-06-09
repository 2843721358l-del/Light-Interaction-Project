#!/usr/bin/env python3
"""
为所有演示视频左下角添加 WASD 键盘 + 摇杆操作指示器。
前 5 秒: W 键高亮（镜头前进）
后 5 秒: S 键高亮（镜头后退）
"""

import os
import subprocess
import sys
import tempfile
import json

from PIL import Image, ImageDraw, ImageFont

# ── 路径配置 ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "static", "assets")
FFMPEG = "/data/cljia/bin/ffmpeg"

VIDEOS = [
    "beach_original.mp4",
    "beach_accelerated.mp4",
    "castle_original.mp4",
    "castle_accelerated.mp4",
]

# ── 视觉设计参数 ───────────────────────────────────────────
# 视频分辨率: 832×480
# 叠加位置: 左下角
OVERLAY_MARGIN_X = 18       # 距左边距离
OVERLAY_MARGIN_BOTTOM = 18  # 距底部距离

# WASD 按键区域
KEY_SIZE = 38               # 每个按键大小
KEY_GAP = 5                 # 按键间距
KEY_RADIUS = 6              # 圆角

# 摇杆区域
JOYSTICK_RADIUS = 32        # 摇杆外圈半径
JOYSTICK_INNER = 12          # 摇杆内圈（拇指）半径
JOYSTICK_GAP = 18           # WASD 和摇杆间距

# 颜色
COLOR_BG = (0, 0, 0, 160)          # 背景底色
COLOR_KEY_DIM = (80, 80, 80, 200)  # 未激活按键
COLOR_KEY_HIGHLIGHT = (118, 185, 0, 240)  # 激活按键 (页面 accent #76b900)
COLOR_KEY_BORDER = (120, 120, 120, 180)     # 按键边框
COLOR_KEY_BORDER_HL = (166, 219, 90, 255)   # 激活按键边框
COLOR_LABEL_DIM = (200, 200, 200, 220)       # 文字暗色
COLOR_LABEL_HL = (255, 255, 255, 255)         # 文字亮色
COLOR_JOY_BG = (30, 30, 30, 180)             # 摇杆背景
COLOR_JOY_BORDER = (100, 100, 100, 160)       # 摇杆边框
COLOR_JOY_DOT = (130, 130, 130, 200)          # 摇杆中心点
COLOR_ARROW_DIM = (90, 90, 90, 160)           # 箭头暗色
COLOR_ARROW_HL = (166, 219, 90, 180)          # 箭头高亮

FONT_SIZE = 14
FONT_SIZE_SMALL = 10

# ── 辅助函数 ────────────────────────────────────────────────

def get_font(size):
    """尝试获取合适的字体"""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-Bold.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


def draw_rounded_rect(draw, xy, radius, fill, outline=None, width=1):
    """绘制圆角矩形"""
    x0, y0, x1, y1 = xy
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def draw_key(draw, cx, cy, key_size, label, is_highlighted):
    """绘制单个按键，返回中心坐标"""
    half = key_size // 2
    x0, y0 = cx - half, cy - half
    x1, y1 = cx + half, cy + half
    fill = COLOR_KEY_HIGHLIGHT if is_highlighted else COLOR_KEY_DIM
    border = COLOR_KEY_BORDER_HL if is_highlighted else COLOR_KEY_BORDER
    label_color = COLOR_LABEL_HL if is_highlighted else COLOR_LABEL_DIM
    draw_rounded_rect(draw, (x0, y0, x1, y1), KEY_RADIUS, fill, border, 2)

    font = get_font(FONT_SIZE)
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((cx - tw // 2, cy - th // 2 - 1), label, fill=label_color, font=font)


def draw_joystick(draw, cx, cy, radius, inner_radius):
    """绘制摇杆（游戏手柄拇指摇杆）"""
    # 外圈
    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        fill=COLOR_JOY_BG,
        outline=COLOR_JOY_BORDER,
        width=2,
    )
    # 内圈（拇指位置）
    draw.ellipse(
        (cx - inner_radius, cy - inner_radius, cx + inner_radius, cy + inner_radius),
        fill=COLOR_JOY_DOT,
        outline=None,
    )
    # 十字指示线
    cross_len = radius - 6
    inner_gap = inner_radius + 3
    for angle in [0, 90, 180, 270]:
        import math
        rad = math.radians(angle)
        x1 = cx + inner_gap * math.cos(rad)
        y1 = cy - inner_gap * math.sin(rad)
        x2 = cx + cross_len * math.cos(rad)
        y2 = cy - cross_len * math.sin(rad)
        draw.line((x1, y1, x2, y2), fill=COLOR_ARROW_DIM, width=2)
    # 外圈边框
    out_r = radius + 1
    draw.arc((cx - out_r, cy - out_r, cx + out_r, cy + out_r), 0, 360,
             fill=COLOR_JOY_BORDER, width=2)


def draw_arrow(draw, cx, cy, direction, size, color):
    """绘制方向箭头三角形"""
    s = size
    if direction == "up":
        pts = [(cx, cy - s), (cx - s * 0.7, cy + s * 0.5), (cx + s * 0.7, cy + s * 0.5)]
    elif direction == "down":
        pts = [(cx, cy + s), (cx - s * 0.7, cy - s * 0.5), (cx + s * 0.7, cy - s * 0.5)]
    elif direction == "left":
        pts = [(cx - s, cy), (cx + s * 0.5, cy - s * 0.7), (cx + s * 0.5, cy + s * 0.7)]
    elif direction == "right":
        pts = [(cx + s, cy), (cx - s * 0.5, cy - s * 0.7), (cx - s * 0.5, cy + s * 0.7)]
    else:
        return
    draw.polygon(pts, fill=color)


def create_overlay(highlight_key=None):
    """
    创建操作指示器叠加图。
    highlight_key: None → 全部暗色; 'W' → W 高亮; 'S' → S 高亮
    返回 RGBA PIL Image
    """
    # 计算 WASD 十字区域尺寸
    wasd_w = KEY_SIZE * 3 + KEY_GAP * 2
    wasd_h = KEY_SIZE * 2 + KEY_GAP

    total_w = wasd_w + OVERLAY_MARGIN_X * 2
    total_h = wasd_h + OVERLAY_MARGIN_BOTTOM * 2

    img = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # ── WASD 按键布局（无背景框）──
    wasd_origin_x = OVERLAY_MARGIN_X
    wasd_origin_y = OVERLAY_MARGIN_BOTTOM

    # 按键中心坐标
    key_cx = wasd_origin_x + KEY_SIZE // 2
    key_cy = wasd_origin_y + KEY_SIZE // 2

    # W: 中上
    w_cx = key_cx + KEY_SIZE + KEY_GAP
    w_cy = key_cy
    # A: 左下
    a_cx = key_cx
    a_cy = key_cy + KEY_SIZE + KEY_GAP
    # S: 中下
    s_cx = w_cx
    s_cy = a_cy
    # D: 右下
    d_cx = key_cx + (KEY_SIZE + KEY_GAP) * 2
    d_cy = a_cy

    draw_key(draw, w_cx, w_cy, KEY_SIZE, "W", highlight_key == "W")
    draw_key(draw, a_cx, a_cy, KEY_SIZE, "A", highlight_key == "A")
    draw_key(draw, s_cx, s_cy, KEY_SIZE, "S", highlight_key == "S")
    draw_key(draw, d_cx, d_cy, KEY_SIZE, "D", highlight_key == "D")

    return img


def get_video_duration(video_path):
    """获取视频时长（秒）"""
    cmd = [
        FFMPEG, "-i", video_path,
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # 从 stderr 中解析 Duration
    for line in result.stderr.split("\n"):
        if "Duration" in line:
            # Duration: 00:03:48.60
            time_str = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = time_str.split(":")
            return float(h) * 3600 + float(m) * 60 + float(s)
    raise ValueError(f"无法获取视频时长: {video_path}")


def process_video(video_name):
    """处理单个视频：生成叠加层并用 ffmpeg 合成"""
    input_path = os.path.join(ASSETS_DIR, video_name)
    output_path = os.path.join(ASSETS_DIR, f"overlay_{video_name}")
    tmp_dir = tempfile.mkdtemp()

    try:
        duration = get_video_duration(input_path)
        print(f"  处理: {video_name} (时长: {duration:.2f}s)")

        # 生成三张叠加图
        base_path = os.path.join(tmp_dir, "base.png")
        w_path = os.path.join(tmp_dir, "w_highlight.png")
        s_path = os.path.join(tmp_dir, "s_highlight.png")

        create_overlay(highlight_key=None).save(base_path)
        create_overlay(highlight_key="W").save(w_path)
        create_overlay(highlight_key="S").save(s_path)

        # ffmpeg 合成命令
        # 三层叠加: base (始终) + W高亮 (t<5) + S高亮 (t>duration-5)
        # 叠加位置: 左下角 x=0, y=H-overlay_h
        overlay_y = f"H-overlay_h-0"

        filter_complex = (
            f"[0][1]overlay=0:{overlay_y}[tmp1];"
            f"[tmp1][2]overlay=0:{overlay_y}:enable='between(t,0,5)'[tmp2];"
            f"[tmp2][3]overlay=0:{overlay_y}:enable='gte(t,{duration-5})'"
        )

        cmd = [
            FFMPEG,
            "-i", input_path,
            "-i", base_path,
            "-i", w_path,
            "-i", s_path,
            "-filter_complex", filter_complex,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-c:a", "copy",
            "-y",
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ✗ FFmpeg 失败:\n{result.stderr[-500:]}")
            return False

        print(f"  ✓ 输出: {output_path}")
        return True

    finally:
        # 清理临时文件
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    print("=" * 60)
    print("  视频操作指示器叠加工具")
    print("  WASD 键盘 → 视频左下角")
    print("  前 5s: W 高亮 | 后 5s: S 高亮")
    print("=" * 60)

    # 从 git 恢复干净原始视频，避免重复叠加
    print("\n🔄 从 git 恢复原始干净视频...")
    video_paths = [os.path.join("static", "assets", v) for v in VIDEOS]
    subprocess.run(
        ["git", "-C", BASE_DIR, "checkout", "HEAD", "--"] + video_paths,
        capture_output=True,
    )
    print("   ✓ 已恢复")

    for video in VIDEOS:
        input_path = os.path.join(ASSETS_DIR, video)
        if not os.path.exists(input_path):
            print(f"  ⚠ 跳过不存在的文件: {input_path}")
            continue
        process_video(video)

    print(f"\n✅ 完成！请手动替换: mv {ASSETS_DIR}/overlay_*.mp4 {ASSETS_DIR}/")


if __name__ == "__main__":
    main()
