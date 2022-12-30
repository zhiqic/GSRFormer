Code is based on https://github.com/jhcho99/CoFormer


## Environment Setup
We provide instructions for environment setup.
```bash
# Clone this repository and navigate into the repository
git clone https://github.com/PYL2077/HiFormer.git
cd GSRFormer

# Create a conda environment, activate the environment and install PyTorch via conda
conda create --name GSRFormer python=3.9              
conda activate GSRFormer
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge

# Install requirements via pip
pip install -r requirements.txt                   
```

## SWiG Dataset
Annotations are given in JSON format, and annotation files are under "SWiG/SWiG_jsons/" directory. Images can be downloaded [here](https://swig-data-weights.s3.us-east-2.amazonaws.com/images_512.zip). Please download the images and store them in "SWiG/images_512/" directory.

#### Additional Details
- All images should be under "SWiG/images_512/" directory.
- train.json file is for train set.
- dev.json file is for development set.
- test.json file is for test set.

## Training
To train GSRFormer on a single node with 4 GPUs, run:
```bash
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
           --backbone resnet50 --dataset_file swig \
		   --encoder_epochs 20 --decoder_epochs 25 \
            --preprocess True \
           --num_workers 4 --num_enc_layers 6 --num_dec_layers 5 \
            --dropout 0.15 --hidden_dim 512 --output_dir GSRFormer
```

- We use AdamW optimizer with learning rate 10<sup>-4</sup> (10<sup>-5</sup> for backbone), weight decay 10<sup>-4</sup> and β = (0.9, 0.999).    
    - Those learning rates are divided by 10 at epoch 30.
- Random Color Jittering, Random Gray Scaling, Random Scaling and Random Horizontal Flipping are used for augmentation.

## Evaluation

```bash
python main.py --output_dir GSRFormer --dev
python main.py --output_dir GSRFormer --test
```

Model Checkpoint can be downloaded [here](https://drive.google.com/u/0/uc?id=1snS2aYo3R-rblQc0Ba7-YZ4mRdhq-6py&export=download)

## Inference
To run an inference on a custom image, run:
```bash
python inference.py --image_path inference/filename.jpg \
                    --output_dir inference
```
