<div align="center">

# PALMIRA

## A Deep Deformable Network for Instance Segmentation of Dense and Uneven Layouts in Handwritten Manuscripts

**_To appear at [ICDAR 2021](https://icdar2021.org/)_**

| **[ [```Paper```](<>) ]** | **[ [```Website```](<https://ihdia.iiit.ac.in/Palmira/>) ]** |
|:-------------------:|:-------------------:|

<br>

<img src="assets/Palmira-Arch-Crop.jpg">

---

</div>

<!-- # Getting the Dataset
> Will be released soon! -->

# Dependencies and Installation

## Manual Setup

The PALMIRA code is tested with

- Python (`3.7.x`)
- PyTorch (`1.7.1`)
- Detectron2 (`0.4`)
- CUDA (`10.0`)
- CudNN (`7.3-CUDA-10.0`)

For setup of Detectron2, please follow
the [official documentation](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

## Automatic Setup (From an Env File)

We have provided environment files for both Conda and Pip methods. Please use any one of the following.

### Using Conda

```bash
conda env create -f environment.yml
```

### Using Pip

```bash
pip install -r requirements.txt
```

# Usage

## Initial Setup:

- Download the Indiscapes-v2 **[[`Dataset Link`](https://github.com/ihdia/indiscapes)]**
- Place the
    - Dataset under `images` directory
    - COCO-Pretrained Model weights in the `init_weights` directory
        - Weights
          used: [[`Mask RCNN R50-FPN-1x Link`](https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl)]
        - Feel free to use other weights from
          Detectron2 [[`Model Zoo`](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md#coco-instance-segmentation-baselines-with-mask-r-cnn)]
        - _NOTE: Pre-trained weights can be turned off in the configs_
    - JSON in `doc_v2` directory More information can be found in folder specific READMEs.

### SLURM Workloads

If your compute uses SLURM workloads, please load these (or equivalent) modules at the start of your experiments. Ensure
that all other modules are unloaded.

```bash
module add cuda/10.0
module add cudnn/7.3-cuda-10.0
```

## Training

### Palmira

Train the presented network

```bash
python train_palmira.py \
    --config-file configs/palmira/Palmira.yaml \
    --num-gpus 4
```

- Any required hyper-parameter changes can be performed in the `Palmira.yaml` file.
- Resuming from checkpoints can be done by adding `--resume` to the above command.

### Ablative Variants and Baselines

Please refer to the [README.md](configs/README.md) under the `configs` directory for ablative variants and baselines.

## Inference

### Quantitative

To perform inference and get quantitative results on the test set.

```bash
python train_palmira.py \
    --config-file configs/palmira/Palmira.yaml \
    --eval-only \
    MODEL.WEIGHTS <path-to-model-file> 
```

### Qualitative

Can be executed only after quantitative inference (or) on validation outputs at the end of each training epoch.

This parses the output JSON and overlays predictions on the images. 

```bash
python visualise_json_results.py \
    --inputs <path-to-output-file-1.json> [... <path-to-output-file-2.json>] \
    --output outputs/qualitative/ \
    --dataset indiscapes_test
```

> NOTE: To compare multiple models, multiple input JSON files can be passed. This produces a single 
> vertically stitched image combining the predictions of each JSON passed.

### Custom Images

To run the model on your own images without training, please download the provided weights.
```bash
python demo.py \
    --input <path-to-image-directory-*.jpg> \
    --output <path-to-output-directory> \
    --config configs/palmira/Palmira.yaml \
    --opts MODEL.WEIGHTS <init-weights.pth>
```

# Citation

```bibtex
@inproceedings{sharan2021palmira,
    title = {PALMIRA: A Deep Deformable Network for Instance Segmentation of Dense and Uneven Layouts in Handwritten Manuscripts},
    author = {Sharan, S P and Aitha, Sowmya and Amandeep, Kumar and Trivedi, Abhishek and Augustine, Aaron and Sarvadevabhatla, Ravi Kiran},
    booktitle = {International Conference on Document Analysis Recognition, {ICDAR} 2021},
    year = {2021},
}
```

# Contact

For any queries, please contact [Dr. Ravi Kiran Sarvadevabhatla](mailto:ravi.kiran@iiit.ac.in.)

# License

This project is open sourced under [MIT License](LICENSE).