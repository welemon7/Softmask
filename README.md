# Illumination-aware Softmask Guided Shadow Removal

Under Review...


## 🧠 Method


## 📊 Results
#### Evaluation on WRSD+
The evaluation results on WRSD+ are as follows
| Method | PSNR | SSIM | RMSE |
| :-- | :--: | :--: | :--: |
| Input Image | 18.87 | 0.825 | 14.76 |
| ShadowFormer | 24.16 | 0.899 | 8.44 |
| HomoFormer | 25.85 | 0.913 | 6.99 |
| **Ours** | **26.29** | **0.923** | **6.63** |

#### Visual Results
<p align="center">
  <img src="./figure/WRSD+.jpg" width="700"/>
</p>

#### Testing results
The testing results on dataset WRSD+: [link](https://drive.google.com/drive/folders/1hW1tVF2JSNqEDEKIgFF4wtrJA8vVymDO?usp=sharing)

## 🛠️ Requirements
```
Python	3.8
PyTorch	2.7.1
CUDA	12.8
```

## 📂 Project Structure


## ▶️ Usage

### 🏋️ Train
1. Download datasets 
```
|-- WRSD+ Dataset
    |-- train
        |-- shadow # shadow image
        |-- non_shadow # shadow-free GT
    |-- test
        |-- shadow # shadow image
        |-- non_shadow # shadow-free GT
```
2. You can to modify the following terms in `option.py`
```python
train_dir  # training set path
test_dir   # testing set path
softmask_dir # testing set path
gpu: 0 # Our model can be trained using RTX 4090 GPU. You can also input "CUDA_VISIBLE_DEVICES=0,1 python train.py".
```
3. Train the network
```bash
python train_softmask.py
python train.py 
```

### 🖊️ Test
```bash
python test_softmask.py
python test.py
```

### 📥 Dataset

Please download datasets from:

* ISTD+ [[link](https://github.com/cvlab-stonybrook/SID)]

* ISTD [[link](https://github.com/DeepInsight-PCALab/ST-CGAN)]

* SRD 

* WRSD+

## 🙏 Acknowledgement

Thanks to previous shadow removal works.

## 📧 Contact

For any questions, please open an issue or contact: [236004855@nbu.edu.cn](mailto:236004855@nbu.edu.cn)