# Image-Super-Resolution-Using-CNN

## Step 1: Prepare the dataset
DIV2K


## Step 2: Train the model
```bash
python -m bin.main train --model SRCNN --model_dir models/SRCNN.model 
```
## Step 3: Test the model
```bash
python -m bin.main eval --model SRCNN --model_dir models/SRCNN.model 
```
