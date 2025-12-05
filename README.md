# 🪿 Cross-GOOSe
**Cross**ing **G**radient-flows for **O**verlapping **O**bjects **Se**gmentation

implementation of _Improving Gradient Flow methods for instance segmentation of crossing objects_ J. Mabon & J.C. Olivo-Marin, submitted to ISBI 2026

## Installation
Setup the environment with [conda/mamba](https://github.com/conda-forge/miniforge) :
```bash
mamba create -f env.yaml -y
mamba activate crossgoose
```

## Training

### preparing the data
```bash
# get the data
mkdir -p data/BBBC010_v2_images 
wget "https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v2_images.zip" -O images.zip
unzip images.zip -d data/BBBC010_v2_images
rm images.zip

wget "https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v1_foreground_eachworm.zip" -O labels.zip
unzip labels.zip -d data
rm labels.zip

# make a dataset
python main.py make_dataset --config configs/dataset.yaml
# make a dataset with synthetic data
python main.py make_synth_dataset --config configs/synth_dataset.yaml

```


### train

```bash
python train.py fit --config configs/model.yaml
```


### Infer/eval
```bash
python main.py eval_models --config configs/eval.yaml
```