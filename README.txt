# DiamondGAN
Tensorflow implementation of DiamondGAN. 

The pre-trained generator is provided, which is trained to translate the MRI brain from T1&amp;T2 to FLAIR&amp;DIR.
![DiamondGAN](https://github.com/dongliangcao/diamondGAN/blob/main/diamondGAN.png)

## Requirement
numpy

tensorflow

tensorflow_addons

SimpleITK

## Usage
### Command line
python model.py --input_dir INPUT_DIR --output_dir OUTPUT_DIR
- The *INPUT_DIR* contains a collection of directories. Each directory should contain **t1.nii.gz**, **t1_bet_mask.nii.gz**, **t2.nii.gz**.
- The generated FLAIR and DIR images would store in the *OUTPUT_DIR* and named as **syn_flair.nii.gz** and **syn_dir.nii.gz**
### Python
from DiamondGAN.model import DiamondGAN

DiamondGAN(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR)

## Notice
If you download the repository through **PyPI**, you may need to download the pre-trained model under the [link](https://drive.google.com/file/d/1BkBc-_yTabEOf1_HJxNjccV9kdg5Dgu5/view) and put it under the If you download the repository through **PyPI**, you may need to download the pre-trained model under the [link](https://drive.google.com/file/d/1BkBc-_yTabEOf1_HJxNjccV9kdg5Dgu5/view) and put it under the **sys.prefix** 
