<!-- Hugging Face Paper 배지 -->
[![Hugging Face Paper](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-blue)](https://huggingface.co/papers/2607.11088)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/jsookim/CUST)

[![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/abs/2607.11088)

# [ECCV 2026] CUST : Clustered Unit-level Similarity Transformer for Lightweight Image Super-Resolution

Author : Jeongsoo Kim


Our project has been accepted as a poster presentation at ECCV 2026.

Our paper is available at [here](https://arxiv.org/abs/2607.11088).

## Requirements

```
# Install Packages
pip install -r requirements.txt
pip install matplotlib

# Install BasicSR
python3 setup.py develop
```


## Dataset

We use DIV2K as Training dataset.
You can download the dataset at https://github.com/dslisleedh/Download_df2k/blob/main/download_df2k.sh
and prepare other test datasets at https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md#Common-Image-SR-Datasets

And also, you'd better extract subimages using 
```
python3 scripts/data_preparation/extract_subimages.py
```
By running the code above, you may get subimages of training datasets.

## Pretrained Models

Pre-trained models can be downloaded from ```experiments/pretrained_model```.

## Training and Test

You can train our CUST following commands below 
```
python3 basicsr/train.py -opt options/train/CUST/cust_base(plus, small)_x2(3,4).yml
```


### Test
You can test our CUST following commands below
```
python3 basicsr/test.py -opt options/test/CUST_base(small)/test_base(small)_benchmark_x2(3, 4).yml
```

## Results
### Result Table with #Param and #FLOPs
![Readme1](https://github.com/user-attachments/assets/664d700d-59a1-43e1-b6ab-9cefc9a1107a)


### Result Table with GPU Consumption and AVG Inference Time
![image](https://github.com/user-attachments/assets/fa179efd-698e-4547-b8b6-356e9b6ab304)

### Qualtitative Results
![image](https://github.com/user-attachments/assets/8796100d-76dd-4fa2-8822-873f07d37b1e)


## Inference Results
We will provide visual results of CUST_Base soon. 
If you want to see only architecture, please refer to `CUST_arch.py`.
