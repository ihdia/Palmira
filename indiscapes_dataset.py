import json
import os

import cv2
import numpy as np
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode

images_root = "images"  # This must contain 4 folders each for a dataset
via_json_root = "doc_v2"  # This must contain 3 folders named train, test, val; each with a via_region_data.json


categories_dict = {
    "Hole(Virtual)": 0,
    "Hole(Physical)": 1,
    "Character Line Segment": 2,
    "Boundary Line": 3,
    "Physical Degradation": 4,
    "Page Boundary": 5,
    "Character Component": 6,
    "Library Marker": 7,
    "Picture / Decorator": 8,
    # In case you are using it on Dataset-v1, please split Pic/Deco into 2 individual classes
}

categories_list = [
    "Hole(Virtual)",
    "Hole(Physical)",
    "Character Line Segment",
    "Boundary Line",
    "Physical Degradation",
    "Page Boundary",
    "Character Component",
    "Library Marker",
    "Picture / Decorator",
]


def get_indiscapes_dicts(img_dir, doc_dir):
    """
    https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html
    Cleans the dataset (file name regularization, etc.) and pre-proccesses to match
    COCO type annotations to be natively loaded into detectron2
    :param img_dir: directory holding images
    :param doc_dir: directory holding annotation jsons
    :return: dictionary of COCO-type formatted annotations of Indiscapes-v2
    """
    json_file = os.path.join(doc_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)
    dataset_dicts = []
    for idx, v in enumerate(imgs_anns["_via_img_metadata"].values()):
        record = {}
        url = v["filename"]
        filename = url.replace("%20", " ")

        bhoomi = ["Bhoomi_data", "bhoomi"]
        if any(x in filename for x in bhoomi):
            if "images" in filename:
                f = filename.split("images")[1]
                file_name1 = img_dir + "/Bhoomi_data/images/images" + f
            else:
                f = filename.split("bhoomi")[1]
                file_name1 = img_dir + "/Bhoomi_data/images/images" + f
        collections = ["penn_in_hand", "penn-in-hand", "jain-mscripts", "ASR_Images"]
        if any(x in filename for x in collections):
            if "penn_in_hand" in filename:
                file_name1 = img_dir + filename.split("9006")[1]
            else:
                file_name1 = img_dir + filename.split("imgdata")[1]

        height, width = cv2.imread(file_name1).shape[:2]
        record["file_name"] = file_name1
        record["height"] = height
        record["width"] = width
        record["image_id"] = idx

        annos = v["regions"]
        objs = []
        for idx, anno in enumerate(annos):
            shape = anno["shape_attributes"]

            if "all_points_x" not in shape.keys():
                shape["all_points_x"] = [
                    shape["x"] - (shape["width"] / 2),
                    shape["x"] - (shape["width"] / 2),
                    shape["x"] + (shape["width"] / 2),
                    shape["x"] + (shape["width"] / 2),
                ]
                shape["all_points_y"] = [
                    shape["y"] + (shape["height"] / 2),
                    shape["y"] - (shape["height"] / 2),
                    shape["y"] - (shape["height"] / 2),
                    shape["y"] + (shape["height"] / 2),
                ]

            px = shape["all_points_x"]
            py = shape["all_points_y"]

            if len(px) < 6:
                while len(px) < 6:
                    px.insert(1, (px[0] + px[1]) / 2)
                    py.insert(1, (py[0] + py[1]) / 2)

            poly = [(x, y) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            region = anno["region_attributes"]["Spatial Annotation"]
            if type(region) is list:
                region = region[0]
            if region == "Decorator":
                region = "Picture / Decorator"
            if region == "Picture":
                region = "Picture / Decorator"
            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": categories_dict[region],
            }
            objs.append(obj)

        record["annotations"] = objs
        if len(objs):
            dataset_dicts.append(record)
    return dataset_dicts


def register_dataset(combined_train_val=False):
    """
    Registers the datasets to DatasetCatalog of Detectron2
    This is to ensure that Detectron can load our dataset from it's configs
    :param combined_train_val: Optional parameter for combining train and validation
    set
    :return: None
    """
    DatasetCatalog.clear()
    for d in ["train_val_combined", "val", "test"] if combined_train_val else ["train", "val", "test"]:
        DatasetCatalog.register(
            "indiscapes_" + d,
            lambda d=d: get_indiscapes_dicts(images_root, os.path.join(via_json_root, d)),
        )
        MetadataCatalog.get("indiscapes_" + d).set(thing_classes=categories_list)
        MetadataCatalog.get("indiscapes_" + d).set(evaluator_type="indiscapes")
