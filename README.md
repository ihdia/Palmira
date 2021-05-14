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
The PALMIRA code is tested with 
- Python 3.7
- PyTorch 1.7.1
- Detectron2
- CUDA 10.0
- CudNN/7.3-CUDA-10.0

For setup of detectron2, please follow the [official documentation](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

# Usage
## Initial Setup:
- Download the Indiscapes-v2 from this [link](),model weights from this [link]() and Ground truth annotations(jsons) from here[link]().
- Place Indiscapes2 in images directory,model weights in the init_weights directory and jsons in doc_v2 directory.
- Setup the Virtual Environment with requirement.txt file
```bash
python -m pip install -r requirements.txt
```
- Load the required cuda modules for either Training or inference.
```bash
   module add cuda/10.0;
   module add cudnn/7.3-cuda-10.0;
```
## Training
### Train Palmira:
```bash
Python train_palmira.py --config-file  configs/palmira/Palmira.yaml num-gpus 4 --resume
```
### Train Deconv:
Remove defgrid mask head by commenting out the `add_defgrid_maskhead_config(cfg)` from the basic setup in `train_net_palmira.py`

```bash
Python train_palmira.py --config-file  configs/dconv/dconv_c3-c5.yaml num-gpus 4 --resume
```

### Train MaskRCNN:
Defgrid mask needs to be removed here as well.
```bash
Python train_palmira.py --config-file  configs/mrcnn/vanilla_mrcnn.yaml num-gpus 4 --resume
```

Logs and other output files can be checked in the outputs directory once the training starts.
## Inference
## Quantitative
To start Inference and get Quantitative results on the test set:
```bash
Python train_palmira.py --config-file  configs/palmira/Palmira.yaml --eval-only --MODEL.WEIGHTS init_weights/defgrid_dconv/defgrid_dconv.pth 
```
## Qualitative - Parsing .json and overlays images
To get Qualititative results of the test dataset ( Images overlaid with output instances )
```bash
python visualise_json_results.py --inputs
 path/to/output_file1.json path/to/output_file2.json --output outputs/qualitative/ --dataset indiscapes_test --conf-threshold 0.5
```
If Mulitple models need to be compared Qualitatively then multiple json files need to be given as input.

## To try it on ur own images:
```bash
python demo.py --input path/to/image_directory/*.jpg --output path/to/output_directory --config configs/palmira/Palmira.yaml  --opts MODEL.WEIGHTS init_weights/defgrid_dconv/defgrid_dconv.pth
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
This project is open sourced under MIT license.