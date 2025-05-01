# EZ-CLIP: Efficient Zero-Shot Video Action Recognition

**Official PyTorch implementation of EZ-CLIP: Efficient Zero-Shot Video Action Recognition**  
[[arXiv]](https://arxiv.org/abs/2312.08010) | [[Published in TMLR 2025]](https://openreview.net/forum?id=xxxx) | [[New Repository: T2L]](https://github.com/Shahzadnit/T2L.git)

## News
🎉 **EZ-CLIP has been published in Transactions on Machine Learning Research (TMLR) 2025** under the title **"T2L: Efficient Zero-Shot Action Recognition with Temporal Token Learning"**.  
Please visit our new repository for the updated codebase and resources: [T2L Repository](https://github.com/Shahzadnit/T2L.git).  
This repository remains available for reference but may not receive further updates.

## Updates
- **Trained model download link**: [Google Drive](https://drive.google.com/drive/folders/1OPt5cXSx-1u_hRXSpst94gMJ5P-c7uBS?usp=sharing).
- **Published paper**: Updated details and new repository for T2L (see above).

## Overview

![EZ-CLIP](EZ-CLIP.png)

EZ-CLIP is a simple and efficient adaptation of CLIP designed for zero-shot video action recognition. It leverages **temporal visual prompting** for seamless temporal adaptation, preserving CLIP's generalization abilities without fundamental architectural changes. Additionally, a novel learning objective guides temporal visual prompts to focus on capturing motion, enhancing learning from video data.

## Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Model Zoo](#model-zoo)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Testing](#testing)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction
EZ-CLIP addresses the challenges of adapting CLIP for zero-shot video action recognition. By introducing **temporal visual prompting** and a motion-focused learning objective, EZ-CLIP achieves efficient and effective performance on video data while maintaining CLIP's robust generalization capabilities. The approach requires no fundamental changes to the core CLIP architecture, making it lightweight and practical.

For the latest advancements, refer to our published work, **T2L: Efficient Zero-Shot Action Recognition with Temporal Token Learning**, in TMLR 2025.

## Prerequisites
To set up the environment, install the required libraries using the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Model Zoo
**NOTE**: All models use the publicly available ViT/B-16-based CLIP model.

### Zero-Shot Results
Models are trained on Kinetics-400 and evaluated directly on downstream datasets.

| Model                | Input  | HMDB-51 | UCF-101 | Kinetics-600 | Model Link                                                                 |
|----------------------|--------|---------|---------|--------------|---------------------------------------------------------------------------|
| EZ-CLIP (ViT-16)     | 8x224  | 52.9    | 79.1    | 70.1         | [Link](https://drive.google.com/file/d/19QNGgaZjPyq0yz7XJGFccS7MV09KMY_K/view?usp=drive_link) |

### Base-to-Novel Generalization Results
Datasets are split into base and novel classes. Models are trained on base classes and evaluated on both.

| Dataset   | Input  | Base Acc. | Novel Acc. | HM   | Model Link                                                                 |
|-----------|--------|-----------|------------|------|---------------------------------------------------------------------------|
| K-400     | 8x224  | 73.1      | 60.6       | 66.3 | [Link](https://drive.google.com/file/d/1q8rBkL0QKNTeJJihWkNUwm1eAGH_OY0U/view?usp=sharing) |
| HMDB-51   | 8x224  | 77.0      | 58.2       | 66.3 | [Link](https://drive.google.com/file/d/1hW2i6agAhpyFvoRgPcOki3coQHx-6oWN/view?usp=sharing) |
| UCF-101   | 8x224  | 94.4      | 77.9       | 85.4 | [Link](https://drive.google.com/file/d/16HTxwbqfi1N8BPVjfrvL6F_A4xLNt-zc/view?usp=sharing) |
| SSV2      | 8x224  | 16.6      | 13.3       | 14.8 | [Link](https://drive.google.com/file/d/1EtpET-s634JnHK7n57vrvqNpE7qH_dHq/view?usp=sharing) |

## Data Preparation
Videos must be extracted into frames for efficient reading. Refer to the `Dataset_creation_scripts` directory for preprocessing instructions.  
Supported datasets:
- [Kinetics](https://deepmind.com/research/open-source/open-source-datasets/kinetics/)
- [UCF101](http://crcv.ucf.edu/data/UCF101.php)
- [HMDB51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)

## Training
To train EZ-CLIP, use the following command:
```bash
python train.py --config configs/K-400/k400_train.yaml
```

## Testing
To evaluate a trained model, use:
```bash
python test.py --config configs/ucf101/UCF_zero_shot_testing.yaml
```

## Citation
If you find this code or pre-trained models useful, please cite our papers:

**Published Paper (TMLR 2025)**:
```bibtex
@article{ahmad2025t2l,
  title={T2L: Efficient Zero-Shot Action Recognition with Temporal Token Learning},
  author={Ahmad, Shahzad and Chanda, Sukalpa and Rawat, Yogesh S},
  journal={Transactions on Machine Learning Research},
  year={2025}
}
```

**arXiv Preprint**:
```bibtex
@article{ahmad2023ezclip,
  title={EZ-CLIP: Efficient Zero-Shot Video Action Recognition},
  author={Ahmad, Shahzad and Chanda, Sukalpa and Rawat, Yogesh S},
  journal={arXiv preprint arXiv:2312.08010},
  year={2023}
}
```

## Acknowledgments
This codebase is built upon [ActionCLIP](https://github.com/sallymmx/ActionCLIP). We thank the authors for their contributions.  
For the latest updates and advancements, please refer to the [T2L Repository](https://github.com/Shahzadnit/T2L.git).

---

**Contact**: For questions or issues, please open an issue on this repository or the [T2L Repository](https://github.com/Shahzadnit/T2L.git).

