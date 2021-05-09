import json
import os

import cv2
import numpy as np
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode

images_root = 'images'  # This must contain 2 folders named Bhoomi_data and penn_in_hand
via_json_root = (
    'doc_v1'  # This must contain 3 folders named train, test, val; each with a via_region_data.json
)


categories_dict = {
    'Hole(Virtual)': 0,
    'Hole(Physical)': 1,
    'Character Line Segment': 2,
    'Boundary Line': 3,
    'Physical Degradation': 4,
    'Page Boundary': 5,
    'Character Component': 6,
    'Library Marker': 7,
    'Picture / Decorator': 8,
}

categories_list = [
    'Hole(Virtual)',
    'Hole(Physical)',
    'Character Line Segment',
    'Boundary Line',
    'Physical Degradation',
    'Page Boundary',
    'Character Component',
    'Library Marker',
    'Picture / Decorator',
]


def get_indiscapes_dicts(img_dir, doc_dir):
    json_file = os.path.join(doc_dir, 'via_region_data.json')
    with open(json_file) as f:
        imgs_anns = json.load(f)
    dataset_dicts = []
    for idx, v in enumerate(imgs_anns['_via_img_metadata'].values()):
        record = {}
        url = v['filename']
        # if "asr_images" not in url.lower():
        #     continue
        # whew_files = [
        #     "illustrations/591_1",
        #     "GOML/991/1.jpg",
        #     "OIMYS/2262/20.jpg",
        #     "0878/images/0005.jpg"
        # ]
        filename = url.replace('%20', ' ')

        bhoomi = ['Bhoomi_data', 'bhoomi']
        if any(x in filename for x in bhoomi):
            if 'images' in filename:
                f = filename.split('images')[1]
                file_name1 = img_dir + '/Bhoomi_data/images/images' + f
            else:
                f = filename.split('bhoomi')[1]
                file_name1 = img_dir + '/Bhoomi_data/images/images' + f
        collections = ['penn_in_hand', 'penn-in-hand', 'jain-mscripts', 'ASR_Images']
        if any(x in filename for x in collections):
            if 'penn_in_hand' in filename:
                file_name1 = img_dir + filename.split('9006')[1]
                # if 'images/penn_in_hand/illustrations/280.jpg' in file_name1 \
                #     or 'images/penn_in_hand/illustrations/496.jpg' in file_name1 \
                #     or 'images/penn_in_hand/illustrations/364.jpg' in file_name1:
                #     print("Excluding this troublesome doc cuz of detr")
                    # continue
            else:
                file_name1 = img_dir + filename.split('imgdata')[1]
        
        # if not any(x in file_name1 for x in whew_files):
        #     continue
        
        height, width = cv2.imread(file_name1).shape[:2]
        record['file_name'] = file_name1
        record['height'] = height
        record['width'] = width
        record['image_id'] = idx

        annos = v['regions']
        objs = []
        for idx, anno in enumerate(annos):
            shape = anno['shape_attributes']

            if 'all_points_x' not in shape.keys():
                shape['all_points_x'] = [
                    shape['x'] - (shape['width'] / 2),
                    shape['x'] - (shape['width'] / 2),
                    shape['x'] + (shape['width'] / 2),
                    shape['x'] + (shape['width'] / 2),
                ]
                shape['all_points_y'] = [
                    shape['y'] + (shape['height'] / 2),
                    shape['y'] - (shape['height'] / 2),
                    shape['y'] - (shape['height'] / 2),
                    shape['y'] + (shape['height'] / 2),
                ]

            px = shape['all_points_x']
            py = shape['all_points_y']

            if len(px) < 6:
                # print(record, idx, len(shape["all_points_x"]))
                while len(px) < 6:
                    px.insert(1, (px[0] + px[1]) / 2)
                    py.insert(1, (py[0] + py[1]) / 2)
                # print(record, idx, len(shape["all_points_x"]))

            poly = [(x, y) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            region = anno['region_attributes']['Spatial Annotation']
            if type(region) is list:
                region = region[0]
            if region == 'Decorator':
                region = 'Picture / Decorator'
            if region == 'Picture':
                region = 'Picture / Decorator'
            obj = {
                'bbox': [np.min(px), np.min(py), np.max(px), np.max(py)],
                'bbox_mode': BoxMode.XYXY_ABS,
                'segmentation': [poly],
                'category_id': categories_dict[region],
            }
            objs.append(obj)

        record['annotations'] = objs
        if len(objs):
            dataset_dicts.append(record)
    return dataset_dicts


def register_dataset(combined_train_val=False):
    DatasetCatalog.clear()
    for d in (
        ['train_val_combined', 'val', 'test'] if combined_train_val else ['train', 'val', 'test']
    ):
        DatasetCatalog.register(
            'indiscapes_' + d,
            lambda d=d: get_indiscapes_dicts(images_root, os.path.join(via_json_root, d)),
        )
        MetadataCatalog.get('indiscapes_' + d).set(thing_classes=categories_list)
        MetadataCatalog.get('indiscapes_' + d).set(evaluator_type='indiscapes')
