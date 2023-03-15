# GAMIVAL: Video Quality Prediction on Mobile Cloud Gaming Content

A MATLAB and Python implementation of GAMIng Video Quality EVALuator (GAMIVAL), which is a new gaming-specific no reference video quality assessment model. GAMIVAL achieves superior performance on the new [LIVE-Meta Mobile Cloud Gaming video quality database] (https://live.ece.utexas.edu/research/LIVE-Meta-Mobile-Cloud-Gaming/index.html).

All videos, including training ones and testing ones, have their features (2180 features). The features are extracted first by a two-branch framework, which combines 1156 NSS features with 1024 CNN features. Then a support vector regressor is utilized to learn the feature-to-score mappings. The SVR parameters are optimized via a grid-search on the training set. Take LIVE-Meta-Mobile Cloud Gaming database for example, in the paper, 480 videos were used as training set, and other 120 videos were used as testing set. In application to Meta’s cloud game, we can use 600 videos as training set to gain a regressor as a quality predictor.

## Demos
### NSS Feature Extraction
```
demo_compute_NSS_feats.m
```

### CNN Feature Extraction
```
$ python demo_compute_CNN_feats.py --dataset_name LIVE-Meta-Gaming
```

### Feature Combination
```
combineFeature.m
```

### Evaluation of BVQA Model
```
$ bash run_all_bvqa_regression.sh
```
or
```
$ python evaluate_bvqa_features_regression.py
```

### Training a SVR / linear SVR model
```
$ python train_SVR.py
```

### Predict Quality Score (Testing) via a pretrained SVR / linear SVR model
```
$ python test_SVR.py
```

## Performance
### SRCC / PLCC
| Metrics | SRCC | PLCC |
| ---: | :---: | :---: |
| NIQE | -0.3900 | 0.4581 |
| BRISQUE | 0.7319 | 0.7394 |
| TLVQM | 0.6553 | 0.6889 |
| VIDEVAL | 0.7621 | 0.7763 |
| RAPIQUE | **0.8740** | **0.9039** |
| GAME-VQP | 0.8709 | 0.8882 |
| NDNet-Gaming | 0.8382 | 0.8200 |
| VSFA | **0.9143** | **0.9264** |
| GAMIVAL | **0.9441** | **0.9524** |

### Speed
Speed was evaluated on the feature extraction function in all the algorithms. For GAMIVAL, speed was evaluated on `demo_compute_NSS_feats.m` and `demo_compute_CNN_feats.py` functions.
| Metrics | Platform | Time(sec) |
| ---: | :---: | :---: |
| NIQE | MATLAB | 728 |
| BRISQUE | MATLAB | **205** |
| TLVQM | MATLAB | 588 |
| VIDEVAL | MATLAB | 959 |
| RAPIQUE | MATLAB | **103** |
| GAME-VQP | MATLAB | 2053 |
| NDNet-Gaming | Python, Tensorflow | 779 |
| VSFA | Python, Pytorch | 2385 |
| GAMIVAL | Python, Tensorflow, MATLAB | **201**|

## Citation

If you use this code for your research, please cite the following paper:

[Y.-C. Chen, A. Saha, C. Davis, B. Qui, X. Wang, I. Katsavounidis, and A. C. Bovik, “Gamival : Video quality prediction on mobile cloud gaming content,” *IEEE Signal Processing Letters*, 2023, doi: 10.1109/LSP.2023.3255011.](https://doi.org/10.1109/LSP.2023.3255011)

## Contacts

- Yu-Chih Chen ( berriechen@utexas.edu ) -- Graduate student, Dept. of ECE, UT Austin.
- Avinab Saha ( avinab.saha@utexas.edu ) -- Graduate student, Dept. of ECE, UT Austin.
- Alan C. Bovik ( bovik@ece.utexas.edu ) -- Professor, Dept. of ECE, UT Austin
