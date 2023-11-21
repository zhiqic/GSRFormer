# GSRFormer: Grounded Situation Recognition Transformer with Alternate Semantic Attention Refinement

GSRFormer is a novel approach for grounded situation recognition (GSR) that aims to mimic human-like understanding of visual scenes. While machines can detect objects and classify images well, interpreting the narrative and semantics conveyed in an image remains challenging. 

GSRFormer seeks to advance GSR by modeling not just primary actions, but the associated entities and roles that form a cohesive visual situation.


<div align="center">
   <img width="801" alt="image" src="https://github.com/zhiqic/GSRFormer/assets/65300431/716c7cce-7316-4424-af50-b67cd2b02382">
</div>


## Key Features

- **Alternating Learning Scheme:** Uses an innovative bidirectional learning process between verbs and nouns to ensure a holistic semantic understanding beyond unidirectional interpretations.

- **Pseudo Labeling:** Initially assumes pseudo labels for semantic roles to focus directly on learning intermediate representations from images, avoiding verb ambiguity issues in conventional GSR. 

- **Support Images:** Leverages supplementary images during training to refine verbs using corresponding nouns and vice versa, enhancing generalization.

For more details, please refer to our [ACM Multimedia 2022 paper](https://arxiv.org/abs/2208.08965).

GSRFormer achieves state-of-the-art performance on two benchmark GSR datasets, advancing scene understanding and narrative interpretation capabilities.


## Setup & Installation

### Prerequisites
- Conda
- PyTorch

### Installation Steps
1. Start by cloning the repository:
   ```bash
   git clone https://github.com/zhiqic/GSRFormer.git
   cd GSRFormer
   ```
2. Create and activate the Conda environment:
   ```bash
   conda create --name GSRFormer python=3.9              
   conda activate GSRFormer
   conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
   ```
3. Install the packages required for the project:
   ```bash
   pip install -r requirements.txt                   
   ```

## Dataset: SWiG
The SWiG dataset plays a pivotal role in the model's training and validation:

- Annotations: Found in "SWiG/SWiG_jsons/".
- Images: Download them [here](https://swig-data-weights.s3.us-east-2.amazonaws.com/images_512.zip) and place them in "SWiG/images_512/".

### Directory Structure
- Images: "SWiG/images_512/"
- Training Set: `train.json`
- Development Set: `dev.json`
- Testing Set: `test.json`

## Training
Kickstart the training with:
```bash
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
           --backbone resnet50 --dataset_file swig \
		   --encoder_epochs 20 --decoder_epochs 25 \
           --preprocess True \
           --num_workers 4 --num_enc_layers 6 --num_dec_layers 5 \
           --dropout 0.15 --hidden_dim 512 --output_dir GSRFormer
```

## Evaluation
Assess your model using:
```bash
python main.py --output_dir GSRFormer --dev
python main.py --output_dir GSRFormer --test
```

## Inference
For real-time application on custom images:
```bash
python inference.py --image_path inference/filename.jpg \
                    --output_dir inference
```
## Acknowledgments

Thank you to the authors of [CoFormer repository](https://github.com/jhcho99/CoFormer) for providing an excellent codebase that enabled our advancements. We sincerely appreciate the support of Microsoft Research throughout this project.


## Citation
Enriching the AI community is our goal. If building upon this work, please reference:
```
@inproceedings{cheng2022gsrformer,
  title={GSRFormer: Grounded Situation Recognition Transformer with Alternate Semantic Attention Refinement},
  author={Cheng, Zhi-Qi and Dai, Qi and Li, Siyao and Mitamura, Teruko and Hauptmann, Alexander},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={3272--3281},
  year={2022}
}
```

## License
Refer to the Apache 2.0 license provided in [LICENSE](LICENSE) for usage details.
