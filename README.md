Sure! Let's streamline and enhance the README to make it more concise and user-friendly.

---

# GSRFormer: Grounded Situation Recognition Transformer with Alternate Semantic Attention Refinement

**Grounded Situation Recognition (GSR)** aims to generate structured semantic summaries of images, capturing "human-like" event understanding. The GSRFormer model achieves state-of-the-art accuracy on the SWiG dataset. This is its official PyTorch implementation.

![Overview Image](https://user-images.githubusercontent.com/65300431/210104895-7f1c8121-9439-494d-b3cf-12217940564c.png)

## Introduction
Traditional GSR methods often follow a two-step approach: 
1. Detect the activity verb.
2. Predict semantic roles based on the detected verb.

However, these methods suffer from some shortcomings, such as not distinguishing between similar daily activities or failing to exploit semantic relationships among verbs and roles.

GSRFormer addresses these issues with a new framework:
1. It assumes a pseudo label and learns intermediate representations for semantic roles.
2. Uses transformer layers to unveil semantic relations, with an alternate learning scheme to optimize the verb and nouns.

![Model Structure](https://user-images.githubusercontent.com/65300431/210105753-7db461da-fb2c-44a4-b9e1-efa1b996986e.png)

## Setup & Installation

### Prerequisites
- Conda (for environment management)
- PyTorch

### Steps
1. Clone the repository:
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
3. Install required packages:
   ```bash
   pip install -r requirements.txt                   
   ```

## Dataset: SWiG
- Annotations: Located in the "SWiG/SWiG_jsons/" directory.
- Images: Download [here](https://swig-data-weights.s3.us-east-2.amazonaws.com/images_512.zip) and store in "SWiG/images_512/" directory.

### Structure
- All images: "SWiG/images_512/" directory.
- Training: `train.json`
- Development: `dev.json`
- Testing: `test.json`

## Training
Train GSRFormer with the following command:
```bash
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
           --backbone resnet50 --dataset_file swig \
		   --encoder_epochs 20 --decoder_epochs 25 \
           --preprocess True \
           --num_workers 4 --num_enc_layers 6 --num_dec_layers 5 \
           --dropout 0.15 --hidden_dim 512 --output_dir GSRFormer
```

## Evaluation
Evaluate using these commands:
```bash
python main.py --output_dir GSRFormer --dev
python main.py --output_dir GSRFormer --test
```

## Inference
For custom image inference:
```bash
python inference.py --image_path inference/filename.jpg \
                    --output_dir inference
```

## Citation
If using this work, please cite:
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
Released under the Apache 2.0 license. Refer to [LICENSE](LICENSE) for details.
