# Convolutional Neural Networks for Crack Classification

## Installation (Windows)

```powershell
virtualenv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Linux

```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```
python run.py -m shallow -l wandb
python run.py -m lenet -l wandb
python run.py -m alexnet -l wandb
```

## Dataset

- <https://www.kaggle.com/datasets/arnavr10880/concrete-crack-images-for-classification>

## Reference

- <https://medium.com/@deepeshdeepakdd2/lenet-5-implementation-on-mnist-in-pytorch-c6f2ee306e37>
- <https://blog.paperspace.com/writing-lenet5-from-scratch-in-python/>
- <https://pytorch.org/tutorials/beginner/data_loading_tutorial.html>
- <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files>
- <https://medium.com/jun94-devpblog/pytorch-1-transform-imagefolder-dataloader-7f75f0a460c0>
- <https://debuggercafe.com/pytorch-imagefolder-for-training-cnn-models/>
- <https://towardsdatascience.com/binary-image-classification-in-pytorch-5adf64f8c781>
- <https://www.analyticsvidhya.com/blog/2021/06/binary-image-classifier-using-pytorch/>
- <https://machinelearningmastery.com/building-a-binary-classification-model-in-pytorch/>
- <https://pytorch.org/vision/main/models/generated/torchvision.models.alexnet.html>
