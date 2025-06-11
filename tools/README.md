## ✅ How to use:

``` bash
python generate_ffmpeg_commands.py \
  --games game1 game2 \
  --res 720x1280 1080x1920 \
  --input_dir ./raw_videos \
  --output_dir ./yuv_videos
```
It will output a run_ffmpeg.sh file with executable FFmpeg commands.

# 📺 YUV Frame Viewer

This tool provides a simple way to **visualize a single frame from a raw YUV 4:2:0 video file**. Useful for debugging decoded video sequences or checking the quality of raw video data.

---

## 🧩 Features

- Supports YUV 4:2:0 format (common for video codecs)
- Displays a selected frame using `matplotlib`
- CLI version for scripting and automation
- Jupyter Notebook version for interactive exploration

---

## 📁 Files

| File                  | Description                                   |
|-----------------------|-----------------------------------------------|
| `read_yuv.py`         | CLI tool to visualize a YUV frame             |
| `read_yuv_notebook.ipynb` | Jupyter notebook version with interactive input |

---

## 🛠️ Usage

### ▶️ CLI Version

```bash
python read_yuv.py --input path/to/video.yuv --width 1920 --height 1080 --frame 10
```

### Parameters:
- input: path to the YUV file
- width: width of the video frame
- height: height of the video frame
- frame: frame index (starting from 0)

## 🧪 Notebook Version
Open read_yuv_notebook.ipynb in Jupyter or VSCode and modify the parameters directly in the cell:

```
input_file = "path/to/video.yuv"
width = 1920
height = 1080
frame_num = 10
```
## 📌 Notes
- This viewer assumes the YUV 4:2:0 planar format with no headers.
- It only reads uncompressed .yuv files (i.e., raw video).
