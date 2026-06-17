# 视频操作指示器(HUD)叠加工具

为视频添加 WASD 按键 + 方向遥感罗盘的操作指示器（HUD）。

## 文件说明

| 文件 | 说明 |
|------|------|
| `add_hud.py` | 主程序，处理单个视频 |

## 基本用法

```bash
python3 add_hud.py 输入视频.mp4 输出视频.mp4 --preset 预设名 --type 视频类型
```

### 参数

| 参数 | 说明 |
|------|------|
| `input` | 输入视频路径 |
| `output` | 输出视频路径 |
| `--preset` | 时序预设（必选）|
| `--type` | 视频类型：`worldplay` 或 `matrix`（默认 worldplay）|
| `--shift` | 手动指定水平偏移像素（覆盖 --type）|

### --type 的区别

- **worldplay**（848×480 等）：HUD 无偏移，默认位置
- **matrix**（1280×704 等）：HUD 整体向右偏移 20px，因为画面比例不同

## 内置时序预设

| 预设名 | 时序 | 适用场景 |
|--------|------|---------|
| `beach` | **W**(0-5s) → **S**(5-10.54s) | WorldPlay 海滩前后 |
| `lake` | 遥感 **left**(0-5s) → **right**(5-10.54s) | WorldPlay 湖泊左右 |
| `castle` | **W**(0-10.41s) → **S**(10.41-19.82s) | Matrix 城堡 |
| `snow` | 遥感 **left**(0-5.7s) → **right**(5.7-10.41s) | Matrix 雪地 |
| `hero` | **W**→遥感左→**W**→遥感右→**W**（127潜在帧） | 首页 2×2 视频 |
| `ad` | **A**(0-5.7s) → **D**(5.7-10.41s) | 通用 A→D |
| `rl` | 遥感 **right**(0-5.2s) → **left**(5.2-10.54s) | 通用 右→左 |
| `lr` | 遥感 **left**(0-5s) → **right**(5-10.54s) | 通用 左→右 |

## 示例

```bash
# WorldPlay 海滩（前后视角）
python3 add_hud.py input.mp4 output.mp4 --preset beach --type worldplay

# Matrix 城堡
python3 add_hud.py input.mp4 output.mp4 --preset castle --type matrix

# 通用 A→D（指定偏移）
python3 add_hud.py input.mp4 output.mp4 --preset ad --shift 20
```

## 自定义时序

在 `add_hud.py` 中按以下格式添加新的 `make_state_fn(...)` 调用：

```python
# 按键名: "W"/"A"/"S"/"D" 为 WASD 高亮, "left"/"right" 为遥感方向
# 格式: [(按键名, 起始秒, 结束秒), ...]
MY_PRESET = make_state_fn([
    ("W", 0, 3.0),      # 0-3s: W 高亮
    ("left", 3.0, 6.0), # 3-6s: 遥感向左
    ("S", 6.0, 10.0),   # 6-10s: S 高亮
])
```

## 依赖

```bash
pip install opencv-python imageio-ffmpeg numpy
```
