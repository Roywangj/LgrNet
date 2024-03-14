# LgrNet


## Install

```bash
# 1. create a conda virtual environment and activate it
conda create -n lgrnet python=3.7 -y
conda activate lgrnet

# 2. install required libs, pytorch1.7+, torchvision, etc.
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=10.2 -c pytorch -y
pip install cycler einops h5py pyyaml==5.4.1 scikit-learn==0.24.2 scipy tqdm matplotlib==3.4.2

# 3. install CUDA kernels
pip install pointnet2_ops_lib/.
```


## Useage

### Classification ModelNet40
**Train**: The dataset will be automatically downloaded, run following command to train.

By default, it will create a folder named "checkpoints/{modelName}-{msg}-{randomseed}", which includes args.txt, best_checkpoint.pth, last_checkpoint.pth, log.txt, out.txt.
```bash
cd classification_ModelNet40
# train 
python main.py --model LgrNet4_4
# please add other paramemters as you wish.
```







## Acknowledgment

Our implementation is mainly based on the following codebase. We gratefully thank the authors for their wonderful works.

[PointMLP](https://github.com/ma-xu/pointMLP-pytorch)









