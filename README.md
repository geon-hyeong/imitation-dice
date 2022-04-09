# DemoDICE
This repository is the official implementation of [DemoDICE: Offline Imitation Learning with Supplementary Imperfect Demonstrations](https://openreview.net/pdf?id=BrPdX1bDZkQ) (presented at ICLR 2022).

## Installation Guide

### Environment Variables
- Insert the following commands in `~/.bashrc`.
    ```
    export MUJOCO_PY_MJKEY_PATH=$HOME/.mujoco/mjkey.txt
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
    ```


### MuJoCo
- Download MuJoCo version 2.1.0. Save 'mjkey.txt' to '$HOME/.mujoco'.

### Conda Environment
1. Create conda environment and activate it:
```
conda env create -f environment.yml
conda activate imitation-dice
```
2. Install 'd4rl':
```
pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl
```

### How to Run
```
python lfd_mujoco.py \
  --env_id=Hopper \
  --imperfect_dataset_info=(["expert-v2", "random-v2"], [400, 1600]) \
  --alpha=0.05 \
  --grad_reg_coeffs=(10., 1e-4) \
  --batch_size=256 \
  --using_absorbing=True
```

### Bibtex

If you use this code, please cite our paper:
```
@inproceedings{kim2022demodice,
  author    = {Geon-Hyeong Kim and Seokin Seo and Jongmin Lee and Wonseok Jeon and HyeongJoo Hwang and Hongseok Yang and Kee-Eung Kim},
  title     = {DemoDICE: Offline Imitation Learning with Supplementary Imperfect Demonstrations},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2022}
}
```