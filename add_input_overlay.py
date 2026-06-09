#!/usr/bin/env python3
"""
为所有演示视频左下角添加 WASD 键盘操作指示器。
按键大小按视频高度比例计算，确保网页展示时大小一致。
HY-WorldPlay (beach): 前 5s W 高亮, 5s 起 S 高亮
Matrix-Game (castle): 前 10.5s W 高亮, 10.5s 起 S 高亮
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

# 每个视频 W→S 切换时间（秒）
SWITCH_TIME = {
    "beach_original.mp4": 5,
    "beach_accelerated.mp4": 5,
    "castle_original.mp4": 10.5,
    "castle_accelerated.mp4": 10.5,
}

# ── 比例参数（均相对 KEY_SIZE）───────────────────────────
KEY_HEIGHT_RATIO = 0.11     # 按键大小 = 视频高度 × 该比例
GAP_RATIO = 0.13            # 间距比例
RADIUS_RATIO = 0.16         # 圆角比例
FONT_RATIO = 0.38           # 字号比例
MARGIN_RATIO = 0.35         # 边距比例

# ── 颜色 ─────────────────────────────────────────────────
COLOR_KEY_DIM = (80, 80, 80, 200)
COLOR_KEY_HIGHLIGHT = (118, 185, 0, 240)        # 页面 accent #76b900
COLOR_KEY_BORDER = (120, 120, 120, 180)
COLOR_KEY_BORDER_HL = (166, 219, 90, 255)
COLOR_LABEL_DIM = (200, 200, 200, 220)
COLOR_LABEL_HL = (255, 255, 255, 255)


# ── 辅助函数 ─────────────────────────────────────────────

def get_font(size):
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


def get_video_info(video_path):
    """获取视频时长和分辨率"""
    cmd = [FFMPEG, "-i", video_path, "-f", "null", "-"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = None
    width = height = None
    for line in result.stderr.split("\n"):
        if "Duration" in line:
            time_str = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = time_str.split(":")
            duration = float(h) * 3600 + float(m) * 60 + float(s)
        if "Stream" in line and "Video" in line:
            # 提取分辨率: 832x480
            import re
            m = re.search(r'(\d{2,})x(\d{2,})', line)
            if m:
                width, height = int(m.group(1)), int(m.group(2))
    if duration is None:
        raise ValueError(f"无法获取视频时长: {video_path}")
    return duration, width, height


def create_overlay(key_size, key_gap, key_radius, font_size, margin, highlight_key=None):
    """创建 WASD 叠加图，所有尺寸参数化"""
    wasd_w = key_size * 3 + key_gap * 2
    wasd_h = key_size * 2 + key_gap

    total_w = wasd_w + margin * 2
    total_h = wasd_h + margin * 2

    img = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # ── WASD 按键布局 ──
    ox = margin
    oy = margin

    kc = ox + key_size // 2
    kr = key_cy = oy + key_size // 2

    w_cx = kc + key_size + key_gap
    w_cy = kr
    a_cx = kc
    a_cy = kr + key_size + key_gap
    s_cx = w_cx
    s_cy = a_cy
    d_cx = kc + (key_size + key_gap) * 2
    d_cy = a_cy

    font = get_font(font_size)
    for cx, cy, label, hl in [
        (w_cx, w_cy, "W", highlight_key == "W"),
        (a_cx, a_cy, "A", highlight_key == "A"),
        (s_cx, s_cy, "S", highlight_key == "S"),
        (d_cx, d_cy, "D", highlight_key == "D"),
    ]:
        half = key_size // 2
        fill = COLOR_KEY_HIGHLIGHT if hl else COLOR_KEY_DIM
        border = COLOR_KEY_BORDER_HL if hl else COLOR_KEY_BORDER
        label_color = COLOR_LABEL_HL if hl else COLOR_LABEL_DIM
        draw.rounded_rectangle(
            (cx - half, cy - half, cx + half, cy + half),
            radius=key_radius, fill=fill, outline=border, width=2,
        )
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((cx - tw // 2, cy - th // 2 - 1), label, fill=label_color, font=font)

    return img


def process_video(video_name):
    """处理单个视频"""
    input_path = os.path.join(ASSETS_DIR, video_name)
    output_path = os.path.join(ASSETS_DIR, f"overlay_{video_name}")
    tmp_dir = tempfile.mkdtemp()

    switch = SWITCH_TIME.get(video_name, 5)

    try:
        duration, width, height = get_video_info(input_path)

        # 按视频高度计算所有尺寸
        ks = round(height * KEY_HEIGHT_RATIO)
        kg = round(ks * GAP_RATIO)
        kr = round(ks * RADIUS_RATIO)
        fs = round(ks * FONT_RATIO)
        mg = round(ks * MARGIN_RATIO)

        print(f"  处理: {video_name}")
        print(f"    分辨率: {width}x{height}, 时长: {duration:.1f}s, W→S: {switch}s")
        print(f"    按键: {ks}px, 字体: {fs}px")

        w_path = os.path.join(tmp_dir, "w_highlight.png")
        s_path = os.path.join(tmp_dir, "s_highlight.png")

        create_overlay(ks, kg, kr, fs, mg, highlight_key="W").save(w_path)
        create_overlay(ks, kg, kr, fs, mg, highlight_key="S").save(s_path)

        # ffmpeg 合成
        overlay_y = "H-overlay_h-0"
        filter_complex = (
            f"[0][1]overlay=0:{overlay_y}:enable='lt(t,{switch})'[tmp1];"
            f"[tmp1][2]overlay=0:{overlay_y}:enable='gte(t,{switch})'"
        )

        cmd = [
            FFMPEG, "-i", input_path, "-i", w_path, "-i", s_path,
            "-filter_complex", filter_complex,
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "copy", "-y", output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ✗ FFmpeg 失败:\n{result.stderr[-500:]}")
            return False

        print(f"  ✓ 完成")
        return True

    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    print("=" * 60)
    print("  视频操作指示器叠加工具")
    print("  WASD 按键 → 视频左下角")
    print(f"  按键大小 = 视频高度 × {KEY_HEIGHT_RATIO}")
    print("=" * 60)

    # 从 git 初始提交恢复干净原始视频
    print("\n🔄 从 git 恢复原始干净视频...")
    video_paths = [os.path.join("static", "assets", v) for v in VIDEOS]
    subprocess.run(
        ["git", "-C", BASE_DIR, "checkout", "508c639", "--"] + video_paths,
        capture_output=True,
    )
    print("   ✓ 已恢复")

    for video in VIDEOS:
        input_path = os.path.join(ASSETS_DIR, video)
        if not os.path.exists(input_path):
            print(f"  ⚠ 跳过: {input_path}")
            continue
        process_video(video)

    print(f"\n✅ 完成！替换命令:")
    for v in VIDEOS:
        print(f"   mv {ASSETS_DIR}/overlay_{v} {ASSETS_DIR}/{v}")


if __name__ == "__main__":
    main()
