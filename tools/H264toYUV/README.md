# YUV Video Processing Pipeline

This repository provides a complete CLI and notebook-based pipeline to process `.264` H.264-encoded videos into `.yuv` format and generate multiple resolution and compression levels for video quality assessment tasks.

---

## 🔧 Features

- Decode pristine `.264` videos into `.yuv` format (YUV420p) at multiple resolutions.
- Encode `.yuv` videos into `.264` at various compression levels (CRF 20–45).
- Decode encoded `.264` videos back to `.yuv` for analysis or comparison.
- Easily switch input/output paths using command-line arguments.
- Available both as CLI tool and Jupyter notebook.

---

## 📂 File Overview

| File | Purpose |
|------|---------|
| `yuv_processing.py` | Python script to run all three stages via CLI. |
| `yuv_processing_notebook.ipynb` | Jupyter notebook version for interactive use. |

---

## 🧪 Dependencies

- Python ≥ 3.6
- FFmpeg (must be installed and available in PATH)
- `argparse`, `os`, `subprocess`, `glob`

---

## 🚀 CLI Usage

```bash
python yuv_processing.py \
    --input_dir /path/to/pristine_264 \
    --resolutions 1920x1080 1280x720 \
    --crf_levels 20 25 30 35 40 45 \
    --output_dir /path/to/output_root
```

### Arguments

- `--input_dir`: Path to the directory containing pristine `.264` files.
- `--resolutions`: One or more resolutions to downscale to (e.g., `1920x1080 1280x720`).
- `--crf_levels`: CRF values for H.264 encoding (e.g., `20 30 40`).
- `--output_dir`: Root directory to store intermediate `.yuv` and encoded videos.

---

## 📝 Notebook Usage

Open `yuv_processing_notebook.ipynb`, modify the parameters at the top of the notebook, and run each cell in order.

---

## 📁 Output Structure

```
output_root/
├── pristine_yuv420p/
│   ├── 1920x1080/
│   └── 1280x720/
├── encoded/
│   ├── 1920x1080/
│   │   └── 20/25/.../
├── encoded_yuv420p/
│   └── 1920x1080/
│       └── 20/25/.../
```

---

## ✨ License

MIT License.
