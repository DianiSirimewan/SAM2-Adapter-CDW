## SAM2-Adapter-CDW: Class-wise Segmentation of Construction and Demolition Waste
## Environment
This code was implemented with Python 3.9 and PyTorch 2.3.0. You can install all the requirements via:
```bash
pip install -r requirements.txt
```


## Quick Start
1. Download the dataset and split into training, validation and testing.
2. Download the pre-trained [SAM 2(Segment Anything)](https://github.com/facebookresearch/segment-anything-2) and put it in ./pretrained.
3. Training:
```bash
torchrun --nproc_per_node=2 --nnodes=1 train.py --config [CONFIG_PATH]
```

4. Evaluation:
```bash
torchrun --nproc_per_node=2 --nnodes=1 test.py --config [CONFIG_PATH] --model [MODEL_PATH]
```
2. Download trained model weights from this shared [link](https://drive.google.com/drive/folders/16cdnJfr2jb-e_FjMVIF5N_gyhYKvzNJJ?usp=drive_link).

## Dataset

### Construction and Demolition Waste Segmentation
- [CDW-Seg - to be added]

## Acknowledgements

SAM2-Adapter: Evaluating & Adapting Segment Anything 2 in Downstream Tasks: Camouflage, Shadow, Medical Image Segmentation, and More

Paper: https://arxiv.org/abs/2408.04579

Code: https://github.com/tianrun-chen/SAM-Adapter-PyTorch/tree/SAM2-Adapter-for-Segment-Anything-2

## Cite

If you find our work valuable for your research, we kindly ask you to consider citing it.
```

Sirimewan, D., Bazli, M., Raman, S., Mohandes, S. R., Kineber, A. F., & Arashpour, M. (2024).
    Deep learning-based models for environmental management: Recognizing construction, renovation, and demolition waste in-the-wild.
    Journal of environmental management, 351, 119908.

Sirimewan, D., Harandi, M., Peiris, H., & Arashpour, M. (2024).
    Semi-supervised segmentation for construction and demolition waste recognition in-the-wild: Adversarial dual-view networks.
    Resources, Conservation and Recycling, 202, 107399. 

Sirimewan, D., Kunananthaseelan, N., Raman, S., Garcia, R., & Arashpour, M. (2024).
    Optimizing waste handling with interactive AI: Prompt-guided segmentation of construction and demolition waste using computer vision.
    Waste Management, 190, 149-160.

```


