# Alzheimer's Detection - Project

## Overview
A collection of models/weights along with the code to train them.

Dataset is obtained kaggle:

Source: https://www.kaggle.com/datasets/yasserhessein/dataset-alzheimer

## GUI

We have implmented a GUI using `streamlit` which is the file named `detection.py`

To run and open the GUI, run in terminal:

```
streamlit run detection.py
```

## Training

We have provided the code we have used to train and produce the model we used. To train on your own, you just need to run all the cells in the .ipynb file

## Remarks

All models is run using the latest tensorflow and keras or pytorch versions except the transfer learning model.

The transfer learning model has dependencies that is only available with keras <3.0.0 version which when downgraded can be loaded however since other models used the latest version, the transfer learning model cannot be replicated in the GUI
