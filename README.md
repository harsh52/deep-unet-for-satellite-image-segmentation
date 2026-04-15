# Deep U-Net for Satellite Image Segmentation

Deep U-Net architecture for satellite image segmentation — land cover classification from multi-band imagery.

![Segmentation Demo](https://github.com/harsh52/deep-unet-for-satellite-image-segmentation/blob/master/visualization/satelliteseg.gif?raw=true)

## What It Does

Segments satellite images into land cover classes (vegetation, water, urban, etc.) using a U-Net convolutional neural network. Includes a Tkinter GUI for interactive prediction.

## How to Run

```bash
pip install -r requirements.txt
python train_unet.py        # Train the model
python predict_UI.py        # Launch the GUI for predictions
```

## Tech

Python, TensorFlow/Keras, U-Net architecture, Tkinter, GDAL for raster processing

## Screenshots

![Tkinter App](https://github.com/harsh52/deep-unet-for-satellite-image-segmentation/blob/master/visualization/Screenshot%20(6625).png?raw=true)
