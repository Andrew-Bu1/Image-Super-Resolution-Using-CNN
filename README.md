# Image-Super-Resolution-Using-CNN

## Step 1: Prepare the dataset
DIV2K
Set5
Set14
Urban100
BSDS100
Manga109


## Step 2: Train the model
```bash
python -m bin.main train --model SRCNN --model_dir models/SRCNN.model 
```
## Step 3: Test the model
```bash
python -m bin.main eval --model SRCNN --model_dir models/SRCNN.model 
```
