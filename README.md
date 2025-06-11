# GAMIVAL: Video Quality Prediction on Mobile Cloud Gaming Content

GAMIVAL (GAMIng Video Quality EVALuator) is a no-reference video quality assessment model designed for cloud gaming videos. It extracts gaming-specific features in MATLAB and Python and trains a Support Vector Regressor (SVR) to predict perceptual quality. The method is introduced in [IEEE Signal Processing Letters 2023](https://doi.org/10.1109/LSP.2023.3255011).

---

## Repository Structure

- `demo_compute_NSS_feats.m` - MATLAB demo to compute Natural Scene Statistics (NSS) features.
- `demo_compute_CNN_feats.py` - Python script to extract 3D CNN features from video frames.
- `combineFeature.m` - merges NSS and CNN features into a single 2180-D descriptor.
- `train_SVR.py` / `test_SVR.py` - scripts to train or apply the SVR model.
- `evaluate_bvqa_features_regression.py` - evaluation utility used by `run_all_bvqa_regression.sh`.
- `mos_files/` - metadata and MOS labels for supported datasets.
- `feat_files/`, `models/`, `result/` - output folders for features, trained regressors, and results.

## Requirements

- MATLAB with access to `ffmpeg` for reading YUV videos.
- Python 3 with packages listed in `requirements.txt` (`pip install -r requirements.txt`).

## Basic Workflow

1. **Extract NSS Features**
   ```bash
   demo_compute_NSS_feats.m
   ```
2. **Extract CNN Features**
   ```bash
   python demo_compute_CNN_feats.py --dataset_name LIVE-Meta-Gaming
   ```
3. **Combine Features**
   ```bash
   combineFeature.m
   ```
4. **Train or Evaluate**
   ```bash
   bash run_all_bvqa_regression.sh        # batch evaluation
   # or
   python evaluate_bvqa_features_regression.py
   python train_SVR.py                     # train a custom model
   ```
5. **Predict Quality Scores**
   ```bash
   python test_SVR.py
   ```

## Performance

GAMIVAL achieves state-of-the-art accuracy on the LIVE-Meta Mobile Cloud Gaming Database. Example SRCC/PLCC results are shown below.

| Method       | SRCC   | PLCC   |
|-------------:|:------:|:------:|
| NIQE         | -0.390 | 0.458  |
| BRISQUE      | 0.732  | 0.739  |
| TLVQM        | 0.655  | 0.689  |
| VIDEVAL      | 0.762  | 0.776  |
| RAPIQUE      | **0.874** | **0.904** |
| GAME-VQP     | 0.871  | 0.888  |
| NDNet-Gaming | 0.838  | 0.820  |
| VSFA         | **0.914** | **0.926** |
| **GAMIVAL**  | **0.944** | **0.952** |

<p align="center"><img src="/figures/boxplot.png" width="50%"></p>

Average feature extraction runtimes (1080p videos) are listed below.

| Method        | Platform | Time (s) |
|--------------:|:--------:|:--------:|
| NIQE          | MATLAB   | 728 |
| BRISQUE       | MATLAB   | **205** |
| TLVQM         | MATLAB   | 588 |
| VIDEVAL       | MATLAB   | 959 |
| RAPIQUE       | MATLAB   | **103** |
| GAME-VQP      | MATLAB   | 2053 |
| NDNet-Gaming  | Python   | 779 |
| VSFA          | PyTorch  | 2385 |
| **GAMIVAL**   | Python/MATLAB | **201** |

<p align="center"><img src="/figures/time.png" width="50%"></p>

## Citation

If you use this repository, please cite:

```
@ARTICLE{10065464,
  author={Chen, Yu-Chih and Saha, Avinab and Davis, Chase and Qiu, Bo and Wang, Xiaoming and Gowda, Rahul and Katsavounidis, Ioannis and Bovik, Alan C.},
  journal={IEEE Signal Processing Letters},
  title={GAMIVAL: Video Quality Prediction on Mobile Cloud Gaming Content},
  year={2023},
  pages={1-5},
  doi={10.1109/LSP.2023.3255011}
}
```

## Contacts

- Yu-Chih Chen (<berriechen@utexas.edu>) – UT Austin
- Avinab Saha (<avinab.saha@utexas.edu>) – UT Austin
- Alan C. Bovik (<bovik@ece.utexas.edu>) – UT Austin
