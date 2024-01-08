# DASSL-FINETUNING

## Introduction

More people use the [Dassl PyTorch toolbox](https://github.com/KaiyangZhou/Dassl.pytorch) to train machine learning models and reproduce SOTA papers making it quite popular nowadays. However, many of them are not aware on how to do fine-tuning from torchvision PyTorch models using this framework. Thus, this repository extends the [Dassl PyTorch toolbox](https://github.com/KaiyangZhou/Dassl.pytorch) for fine-tuning pre-trained models on custom datasets. It includes implementations for fine-tuning VGG16 and ResNet50 models, serving as examples of how to adapt Dassl for specific fine-tuning tasks.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.8+
- PyTorch 1.8.1+
- CUDA 10.2+ (for GPU support)

## Installation

To set up the fine-tuning environment, follow these steps:

```bash
# Clone this repo
git clone https://github.com/ravialdy/DASSL-FINETUNING.git
cd DASSL-FineTuning/

# Create and activate a conda environment
conda create -y -n dassl-ft python=3.8
conda activate dassl-ft

# Install dependencies
pip install -r requirements.txt

# Install the Dassl library in development mode
cd my_dassl/
python setup.py develop
cd ..
```

## Dataset Preparation

To prepare your datasets, follow the instructions in [datasets/README.md](./datasets/README.md). Example datasets included are SVHN, Oxford Pets, and others.

## Fine-Tuning Models

### Fine-Tuning VGG16

To fine-tune the VGG16 model, use the script in `scripts/ftvgg16/`. For example if the downstream dataset is SVHN,

```bash
cd scripts/ftvgg16
sh base_train_gpu0.sh svhn 1000
```

Here, 1000 refers to the maximum epoch in the traininig process that we want. Let's say the downstream dataset is still SVHN,

### Fine-Tuning ResNet50

To fine-tune the ResNet50 model, use the script in `scripts/ftresnet/`.

```bash
cd scripts/ftresnet
sh base_train_gpu1.sh svhn 1000
```

**Note:** You can change the gpu used for this fine-tuning process. For instance, you just want to utilize GPU 1, then you can change 'CUDA_VISIBLE_DEVICES=0' in the shell script into 'CUDA_VISIBLE_DEVICES=1'.

## Customization

To adapt this framework for other models or datasets, follow the patterns in `trainers/ftvgg16.py` and `trainers/ftresnet50.py`. Implement your dataset handling in `datasets/` and create corresponding configuration files in `configs/`.

## License

This project is licensed under the terms of the [MIT License](LICENSE).

## Acknowledgments

This work builds upon the [Dassl PyTorch toolbox](https://github.com/KaiyangZhou/Dassl.pytorch) developed by Kaiyang Zhou.