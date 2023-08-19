# **GSRFormer**: Grounded Situation Recognition Transformer with Alternate Semantic Attention Refinement

Understanding visual scenes has long been at the forefront of AI research. While machines excel at detecting objects and classifying images, truly interpreting the narrative an image conveys — as humans naturally do — remains challenging. This is the realm of **Grounded Situation Recognition (GSR)**.

**GSRFormer** pushes the boundaries in this field. It seeks to replicate human-like interpretation of visual scenarios, recognizing not just primary actions, but also the underlying entities or roles associated with these actions.

<div align="center">
   <img width="801" alt="image" src="https://github.com/zhiqic/GSRFormer/assets/65300431/716c7cce-7316-4424-af50-b67cd2b02382">
</div>


### Why GSRFormer?
While traditional GSR methods are robust, they sometimes overlook the subtle intricacies that distinguish similar actions and the rich semantic relationships binding them. GSRFormer fills this gap through:


1. **Rethinking Verb Detection:** Contrary to traditional approaches that start by detecting the activity verb, our method initially assumes a pseudo label, delving into learning intermediate representations for associated semantic roles directly from images. This sidesteps the pitfalls of ambiguous verb distinctions seen in conventional methods, enhancing the depth of event understanding.

2. **Leveraging Bidirectional Semantic Relations:** Through an innovative alternating learning scheme, backed by support images, verbs are optimized using image-corresponding nouns, while nouns benefit from verbs extracted from support images. This bidirectional refinement ensures a holistic comprehension of the visual narrative, far surpassing unidirectional interpretations.


<div align="center">
   <img width="963" alt="image" src="https://github.com/zhiqic/GSRFormer/assets/65300431/6180bccb-0c41-4c31-8ada-d95cfea65f2c">
</div>


For a more comprehensive understanding of the approach and its benefits, please take a look at our [ACM Multimedia 2022 paper](https://arxiv.org/abs/2208.08965).


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

First and foremost, we would like to extend our deepest gratitude to the original authors and contributors of the [CoFormer repository](https://github.com/jhcho99/CoFormer). Our work stands on the shoulders of giants, and the foundation laid by their research and codebase was invaluable for the advancements we made. Furthermore, we want to express our sincere appreciation to Microsoft Research for their invaluable support throughout our project.


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

Join us in our journey towards richer, more intuitive machine interpretations of the visual world. Explore GSRFormer today!
