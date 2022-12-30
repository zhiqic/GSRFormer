# [ACM-MM'22] GSRFormer: Grounded Situation Recognition Transformer with Alternate Semantic Attention Refinement
[Paper](https://arxiv.org/abs/2208.08965) | [Model Checkpoint](https://drive.google.com/u/0/uc?id=1snS2aYo3R-rblQc0Ba7-YZ4mRdhq-6py&export=download) 

## Overview
Grounded situation recognition is the task of predicting the main activity, entities playing certain roles within the activity, and bounding-box groundings of the entities in the given image. To effectively deal with this challenging task, we introduce a novel approach where the two processes for activity classification and entity estimation are interactive and complementary. To implement this idea, we propose **Co**llaborative Glance-Gaze Trans**Former** (CoFormer) that consists of two modules: Glance transformer for activity classification and Gaze transformer for entity estimation. Glance transformer predicts the main activity with the help of Gaze transformer that analyzes entities and their relations, while Gaze transformer estimates the grounded entities by focusing only on the entities relevant to the activity predicted by Glance transformer. Our CoFormer achieves the state of the art in all evaluation metrics on the SWiG dataset.

![overall_architecture](https://user-images.githubusercontent.com/55849968/160762199-def33a41-b333-41c8-b367-7b6c814b987c.png)
Following conventions in the literature, we call an activity ***verb*** and an entity ***noun***. Glance transformer predicts a verb with the help of Gaze-Step1 transformer that analyzes nouns and their relations by leveraging role features, while Gaze-Step2 transformer estimates the grounded nouns for the roles associated with the predicted verb. Prediction results are obtained by feed forward networks (FFNs). 

## Environment Setup
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

## Acknowledgements
Our code is modified and adapted from these amazing repositories:
- [End-to-End Object Detection with Transformers](https://github.com/facebookresearch/detr)          
- [Grounded Situation Recognition](https://github.com/allenai/swig)
- [Collaborative Transformers for Grounded Situation Recognition](https://github.com/jhcho99/CoFormer)

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
