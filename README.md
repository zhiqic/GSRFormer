# [ACM MM'22] GSRFormer: Grounded Situation Recognition Transformer with Alternate Semantic Attention Refinement
**Grounded Situation Recognition (GSR)** aims to generate structured semantic summaries of images for **"human-like" event understanding**. Specifically, GSR task not only detects **the salient activity verb (e.g. buying)**, but also predicts all corresponding **semantic roles (e.g. agent and goods)**. 
- This is the official PyTorch implementation of [GSRFormer: Grounded Situation Recognition Transformer with Alternate Semantic Attention Refinement](https://arxiv.org/abs/2208.08965). 
- GSRformer achieves state-of-the-art accuracy on the SWiG dataset. This repository contains instructions, code and model checkpoint.

<p align="center">
<img width="1152" alt="image" src="https://user-images.githubusercontent.com/65300431/210104895-7f1c8121-9439-494d-b3cf-12217940564c.png">
</p>

---

## Overview
Inspired by object detection and image captioning tasks, **existing methods typically employ a two-stage framework**: 1) **detect the activity verb**, and then 2) **predict semantic roles based on the detected verb**. Obviously, this illogical framework constitutes a huge obstacle to semantic understanding. 
- First, pre-detecting verbs solely without semantic roles inevitably **fails to distinguish many similar daily activities (e.g., offering and giving, buying and selling)**. 
- Second, predicting semantic roles in a closed auto-regressive manner can **hardly exploit the semantic relations among the verb and roles**. 

<p align="center">	
<img width="1150" alt="image" src="https://user-images.githubusercontent.com/65300431/210104573-ca8eb615-5197-4a7e-81e3-fb499376132a.png">
</p>

To this end, we propose **a novel two-stage framework that focuses on utilizing such bidirectional relations within verbs and roles**. 
- In the first stage, instead of pre-detecting the verb, we **postpone the detection step and assume a pseudo label**, where an intermediate representation for each corresponding semantic role is learned from images. 
- In the second stage, we **exploit transformer layers to unearth the potential semantic relations within both verbs and semantic roles**. With the help of a set of support images, an alternate learning scheme is designed to simultaneously optimize the results: update the verb using nouns corresponding to the image, and update nouns using verbs from support images. 

<p align="center">	
<img width="1049" alt="image" src="https://user-images.githubusercontent.com/65300431/210105753-7db461da-fb2c-44a4-b9e1-efa1b996986e.png">
</p>
 

## Environment Setup
Our code is modified and adapted from these amazing repositories:
- [End-to-End Object Detection with Transformers](https://github.com/facebookresearch/detr)          
- [Collaborative Transformers for Grounded Situation Recognition](https://github.com/jhcho99/CoFormer)

We provide instructions for environment setup.
```bash
# Clone this repository and navigate into the repository
git clone https://github.com/zhiqic/GSRFormer.git
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

<p align="left">  
<img src="https://user-images.githubusercontent.com/55849968/160765697-fbd65b24-6cdd-4b57-8a68-0746cfb7c44e.png" height="200">  
</p> 

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



## Citation
Please cite our paper:
````BibTeX
@inproceedings{cheng2022gsrformer,
  title={GSRFormer: Grounded Situation Recognition Transformer with Alternate Semantic Attention Refinement},
  author={Cheng, Zhi-Qi and Dai, Qi and Li, Siyao and Mitamura, Teruko and Hauptmann, Alexander},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={3272--3281},
  year={2022}
}
````

## License
GSRFormer is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.

## Acknowledgements
This work was supported by the Air Force Research Laboratory under agreement number FA8750-19-2-0200; the Defense Advanced Research Projects Agency (DARPA) grants funded under the GAILA program (award HR00111990063); the Defense Advanced Research Projects Agency (DARPA) grants funded under the AIDA program~(FA8750-18-20018).

The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright notation thereon. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of the Air Force Research Laboratory or the U.S. Government.
