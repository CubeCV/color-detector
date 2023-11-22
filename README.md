# Color Detector
Object detection on colors in a Rubik's Cube

## Commands

### Initial Setup
```bash
git clone git@github.com:CubeCV/color-detector.git
cd color-detector
conda env create -f env.yml
conda activate color-detector-env
```

### Train
```bash
CUDA_VISIBLE_DEVICES=<GPU ID> python main.py
```

### Validate
```bash
CUDA_VISIBLE_DEVICES=<GPU ID> python validate.py --model=path/to/best.pt
```

The path to best.pt is currently `models/best.pt`.

### Predict
```bash
CUDA_VISIBLE_DEVICES=<GPU ID> python predict.py --model=path/to/best.pt --image=path/to/image
```

The path to best.pt is currently `models/best.pt`.

## Datasets
The data for this project is taken from [https://universe.roboflow.com/psst/mofang-i7ha2](https://universe.roboflow.com/psst/mofang-i7ha2).

We have split it into `train` (1198 images + labels), `val` (100 images + labels) , and `test` (100 images + labels) in the `datasets` directory. There is also a `datasets/custom` directory which contains images not from the aforementioned dataset.

## Models
We used the pretrained `yolov8n.pt` model to build our own color detector model. Our custom model is `models/best.pt`.

## Notes
Our model performs well when the cube is isolated (no background distractions) as can be seen in `runs/detect/predict/`. Hence, we would need to segment the cube prior to running the color detection model, which is phase 1 of the CubeCV project.
