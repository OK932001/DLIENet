

# Low-Light Face Image Enhancement via Multi-Dimensional Perception

## Requirements

```bash
python 3.8.10  
pytorch 1.10.0  
numpy==1.21.0  
opencv-python==4.5.5.64  
scikit-image==0.18.3  
```

## DATA

You can download the low-light face dataset (e.g., LOL-Face, DARK-Face) from the official sources.
Please organize your dataset in the following structure:

```
dataset/
├── train/
│   ├── low/
│   └── normal/
├── val/
│   ├── low/
│   └── normal/
```

## PREPROCESSING

* Use `utils/prepare_dataset.py` to generate necessary training/validation lists or CSV files.
* If using 3D geometry features, extract 3D information using `tools/extract_3d_features.py`.

## TRAINING

Modify the configuration file (`./configs/model=MDPLIENet/config.yaml`) to set your training parameters.

### Train the 3D texture-aware enhancement network:

```bash
python train_Texture3DENet.py ./configs/model=Texture3DENet/config.yaml
```

### Train the 2D structure-guided enhancement network:

```bash
python train_Structure2DENet.py ./configs/model=Structure2DENet/config.yaml
```

### Train the fusion decoder (MDPLIENet):

```bash
python train_MDPLIENet.py ./configs/model=MDPLIENet/config.yaml
```

## Pretrained Models

You can download pretrained models (trained on DARK-Face and LOL-Face) [here](#).
Place them in the `./checkpoints` directory.

## TEST

```bash
python test.py --config ./configs/model=MDPLIENet/config.yaml --weights ./checkpoints/mdplienet.pth
```



