# DiamondGAN
Tensorflow implementation of DiamondGAN. 

The pre-trained generator is provided, which is trained to translate the MRI brain from T1&amp;T2 to FLAIR&amp;DIR.

## Requirement
numpy
tensorflow
SimpleITK

## Usage
python model.py --input_dir INPUT_DIR --output_dir OUTPUT_DIR
- The *INPUT_DIR* contains a collection of directories. Each directory should contain **t1.nii.gz**, **t1_bet_mask.nii.gz**, **t2.nii.gz**.
- The generated FLAIR and DIR images would store in the *OUTPUT_DIR* and named as **syn_flair.nii.gz** and **syn_dir.nii.gz**
