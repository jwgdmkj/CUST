# CUST(Clustered Unit-level Similarity Transformer for Lightweight Image Super-Resolution)

Jeongsoo Kim

## Requirements

```
# Install Packages
pip install -r requirements.txt
pip install matplotlib

# Install BasicSR
python3 setup.py develop
```


## Dataset

We use DIV2 as Training dataset.
You can download two datasets at https://github.com/dslisleedh/Download_df2k/blob/main/download_df2k.sh
and prepare other test datasets at https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md#Common-Image-SR-Datasets

And also, you'd better extract subimages using 
```
python3 scripts/data_preparation/extract_subimages.py
```
By running the code above, you may get subimages of training datasets.



### Training
You can train LMLT following commands below 
```
python3 basicsr/train.py -opt options/train/CUST/cust_base(plus, small)_x2(3,4).yml
```


### Test
You can test LMLT following commands below
```
python3 basicsr/test.py -opt options/test/CUST_base(small)/test_base(small)_benchmark_x2(3, 4).yml
```

### Result
#### Result Table with #Param and #FLOPs
![Readme1](https://github.com/user-attachments/assets/664d700d-59a1-43e1-b6ab-9cefc9a1107a)


#### Result Table with GPU Consumption and AVG Inference Time
![image](https://github.com/user-attachments/assets/fa179efd-698e-4547-b8b6-356e9b6ab304)

#### Qualtitative Results
![image](https://github.com/user-attachments/assets/8796100d-76dd-4fa2-8822-873f07d37b1e)

### Results
We will provide visual results of CUST_Base soon. 
If you want to see only architecture, please refer to `CUST_arch.py`.
