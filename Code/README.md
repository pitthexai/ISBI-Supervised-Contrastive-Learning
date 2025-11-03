## Folder Overview

**`YOLOv11/`**  
Image preprocessing using YOLOv11 for automatic detection and cropping of knee AP-view regions from full X-ray images.

**`resnet_model/`**  
Implements the ResNet-18 model as the CNN baseline.

**`contrastive_learning/`**  
Includes implementations for:  
- **SupCon:** supervised contrastive learning model.  
- **SimCLR:** unsupervised contrastive learning baseline.  
- **XAS-SupCon:** explainable age- and sex-aware supervised contrastive learning model.

**`networks/`**  
Backbone and heads used across the contrastive pipelines.

**`grad_cam.ipynb`**  
Visualizes model attention maps using Grad-CAM++ for interpretability.

---

## Dataset Structure

All datasets follow the structure:

```
root_dir/
  class_1/
    image1.jpg
    image2.jpg
    ...
  class_2/
    image1.jpg
    image2.jpg
    ...
```

---

## Training Commands

### XAS-SupCon (Explainable Age- and Sex-Aware Supervised Contrastive Learning)

**Encoder Training**
```bash
python main_xas_supcon.py --batch_size 64   --learning_rate 0.05   --model resnet18   --epochs 200   --size 112   --temp 0.1   --age_sigma 10.2   --age_weight 0.05   --sex_weight 0.05   --cosine   --dataset path   --data_folder ../data/cropped_data/train_val   --mean "(0.4914, 0.4822, 0.4465)"   --std "(0.2675, 0.2565, 0.2761)"
```

**Linear Classification**
```bash
python main_linear_xas_supcon.py --batch_size 64   --learning_rate 0.005   --size 112   --epochs 30   --model resnet18   --ckpt save/SupCon/path_models/XAS_SupCon/last.pth   --dataset path   --data_folder ../data/cropped_data/   --mean "(0.4914, 0.4822, 0.4465)"   --std "(0.2675, 0.2565, 0.2761)"
```

---

### SupCon (Supervised Contrastive Learning)

**Stage 1 – Encoder Training**
```bash
python main_xas_supcon.py --batch_size 64   --learning_rate 0.05   --model resnet18   --epochs 200   --size 112   --temp 0.1   --cosine   --dataset path   --data_folder ../data/cropped_data/train_val   --mean "(0.4914, 0.4822, 0.4465)"   --std "(0.2675, 0.2565, 0.2761)"
```

**Stage 2 – Linear Classification**
```bash
python main_linear_xas_supcon.py --batch_size 64   --learning_rate 0.005   --size 112   --epochs 30   --model resnet18   --ckpt save/SupCon/path_models/SupCon/last.pth   --dataset path   --data_folder ../data/cropped_data/   --mean "(0.4914, 0.4822, 0.4465)"   --std "(0.2675, 0.2565, 0.2761)"
```

---

### SimCLR (Unsupervised Contrastive Learning)

> The only difference from SupCon is adding `--method SimCLR`.

**Stage 1 – Encoder Training**
```bash
python main_xas_supcon.py --batch_size 64   --learning_rate 0.05   --model resnet18   --epochs 200   --size 112   --temp 0.5   --cosine   --dataset path   --data_folder ../data/cropped_data/train_val   --mean "(0.4914, 0.4822, 0.4465)"   --std "(0.2675, 0.2565, 0.2761)"   --method SimCLR
```

**Stage 2 – Linear Classification**
```bash
python main_linear_xas_supcon.py --batch_size 64   --learning_rate 0.05   --size 112   --epochs 30   --model resnet18   --ckpt save/SupCon/path_models/SimCLR/last.pth   --dataset path   --data_folder ../data/cropped_data/   --mean "(0.4914, 0.4822, 0.4465)"   --std "(0.2675, 0.2565, 0.2761)"
```

---

## Notes

- Ensure data paths are correctly configured relative to the repository root.  
- Training logs, checkpoints, and visualizations are stored in the `save/` and `runs/` directories.  
- YOLOv11 and Grad-CAM++ require **OpenCV**, **Pillow**, and **Ultralytics** packages.  
- For reproducibility, fix random seeds and use consistent normalization parameters.  

---

