"""
Overall HD
This is the original evaluator
"""
import collections

import cv2
import numpy as np
from detectron2.data import DatasetCatalog
from detectron2.evaluation import DatasetEvaluator

from medpy.metric import hd, hd95, assd
from scipy.interpolate import splprep, splev

from palmira.indiscapes_dataset import categories_list


def _proc_annotations(annotations):
    dic = {}
    for file in annotations.copy():
        for region in file['annotations']:
            l = region['segmentation'][0]
            n = 2
            x = [l[i: i + n] for i in range(0, len(l), n)]
            region['segmentation'] = x
        dic[file['file_name']] = file

        segm = dic[file['file_name']]['annotations']
        segm_per_region = {i: [] for i in range(len(categories_list))}
        for region in segm:
            segm_per_region[region['category_id']].append(region['segmentation'])
        dic[file['file_name']]['segm_per_region'] = segm_per_region
    return dic


def get_biggest_contour(contours):
    final_contour = contours[0]
    l = 0
    for c in range(len(contours)):
        clen = contours[c].shape[0]
        if clen > l:
            l = clen
    return contours[c]


def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def downsample_points(output):
    predmasks = output['instances'].pred_masks
    predmasks = (predmasks.cpu()) * 255
    predmasks = np.uint8(predmasks.numpy())

    segm_per_region = {i: [] for i in range(len(categories_list))}

    for i in range(len(output['instances'])):
        _,contours, hierarchy= cv2.findContours(
            predmasks[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) == 0:
            continue
        contours = get_biggest_contour(contours)
        contours = np.squeeze(contours)
        try:
            x = contours[:, 1]
            y = contours[:, 0]
        except:
            if contours.shape == (2,):
                continue
            raise Exception("No x and y???")
        okay = np.where(np.abs(np.diff(x)) + np.abs(np.diff(y)) > 0)
        x = np.r_[x[okay], x[-1], x[0]]
        y = np.r_[y[okay], y[-1], y[0]]
        pred_class = categories_list[output['instances'].pred_classes[i]]
        if pred_class == 'Character Line Segment':
            nbr_of_pts = 50
        else:
            nbr_of_pts = 20
        try:
            tck, u = splprep([x, y], k=1, s=0)
            u_new = np.linspace(u.min(), u.max(), int(nbr_of_pts))
            smoothened = np.zeros((int(nbr_of_pts), 2), dtype=np.float32)
            [smoothened[:, 1], smoothened[:, 0]] = splev(u_new, tck, der=0)
        except:
            raise Exception("Idk wat this is!")
        segm_per_region[output['instances'].pred_classes[i].item()].append(smoothened)
    return segm_per_region


class HDEvaluator(DatasetEvaluator):
    def __init__(self, dataset):
        self.annotations = DatasetCatalog.get(dataset)
        self.annotations = _proc_annotations(self.annotations)
        self.count = 0
        # Average
        self.ahd = {cat: [] for cat in categories_list}
        # Normal
        self.hd = {cat: [] for cat in categories_list}
        # 95th percentile
        self.hd95 = {cat: [] for cat in categories_list}
        # IoU
        self.iou = {cat: [] for cat in categories_list}
        # self._logger = logging.getLogger('detectron2.evaluation.coco_evaluation')
        # Per Pixel Accuracy
        self.acc = {cat: [] for cat in categories_list}
        self.doc_wise = {}

    def reset(self):
        self.count = 0
        self.ahd = {cat: [] for cat in categories_list}
        self.hd = {cat: [] for cat in categories_list}
        self.hd95 = {cat: [] for cat in categories_list}
        self.iou = {cat: [] for cat in categories_list}
        self.acc = {cat: [] for cat in categories_list}
        self.doc_wise = {}

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            self.count += len(output['instances'])
            gt_segm = self.annotations[input['file_name']]['segm_per_region']
            try:
                _ = output['instances'].pred_masks
            except AttributeError:
                continue
            pred_segm = downsample_points(output)
            doc_ahd = {cat: [] for cat in categories_list}
            doc_hd = {cat: [] for cat in categories_list}
            doc_hd95 = {cat: [] for cat in categories_list}
            doc_iou = {cat: [] for cat in categories_list}
            doc_acc = {cat: [] for cat in categories_list}
            for reg_type in range(len(categories_list)):
                gt, pred = gt_segm[reg_type], pred_segm[reg_type]

                # Both have points
                if len(gt) and len(pred):
                    gt_mask = np.zeros((input['height'], input['width']), dtype=np.int8)
                    for i in gt:
                        cv2.fillPoly(gt_mask, np.array([i]).astype(np.int32), 1)
                    pred_mask = np.zeros((input['height'], input['width']), dtype=np.int8)
                    for i in pred:
                        cv2.fillPoly(pred_mask, np.array([i]).astype(np.int32), 1)
                    gt_mask = gt_mask.astype(np.uint8)
                    gt_mask = (gt_mask * 255).astype(np.uint8)
                    pred_mask = pred_mask.astype(np.uint8)
                    pred_mask = (pred_mask * 255).astype(np.uint8)

                    def compute_iou_and_accuracy(arrs, edge_mask1):
                        intersection = cv2.bitwise_and(arrs, edge_mask1)
                        union = cv2.bitwise_or(arrs, edge_mask1)
                        intersection_sum = np.sum(intersection)
                        union_sum = np.sum(union)
                        iou = (intersection_sum) / (union_sum)
                        total = np.sum(arrs)
                        correct_predictions = intersection_sum
                        accuracy = correct_predictions / total
                        # print(iou, accuracy)
                        return iou, accuracy

                    res_iou, res_accuracy = compute_iou_and_accuracy(pred_mask, gt_mask)
                    res_ahd, res_hd, res_hd95 = assd(pred_mask, gt_mask), hd(pred_mask, gt_mask), hd95(pred_mask,
                                                                                                       gt_mask)
                    self.ahd[categories_list[reg_type]].append(res_ahd)
                    self.hd[categories_list[reg_type]].append(res_hd)
                    self.hd95[categories_list[reg_type]].append(res_hd95)
                    self.hd[categories_list[reg_type]].append(hd(pred_mask, gt_mask))
                    self.iou[categories_list[reg_type]].append(res_iou)
                    self.acc[categories_list[reg_type]].append(res_accuracy)

                    doc_ahd[categories_list[reg_type]].append(res_ahd)
                    doc_hd[categories_list[reg_type]].append(res_hd)
                    doc_hd95[categories_list[reg_type]].append(res_hd95)
                    doc_hd[categories_list[reg_type]].append(hd(pred_mask, gt_mask))
                    doc_iou[categories_list[reg_type]].append(res_iou)
                    doc_acc[categories_list[reg_type]].append(res_accuracy)
                # One has points
                # elif len(gt) ^ len(pred):
                #     total_area = 0
                #     for each_pred in pred:
                #         total_area += PolyArea(each_pred[:, 0], each_pred[:, 1])
                #     hd = total_area / 100
                #     self.ahd[categories_list[reg_type]].append(hd)
                #     self.hd[categories_list[reg_type]].append(hd)
                #     self.hd95[categories_list[reg_type]].append(hd)
                # self.iou[categories_list[reg_type]].append(0)
                # self.acc[categories_list[reg_type]].append(0)
                # Both Empty
                # elif len(gt) == 0 and len(pred) != 0:

                else:
                    # self.hd[categories_list[reg_type]].append(0)
                    pass
            # print("Over for doc")
            total_ahd = list()
            for l in doc_ahd.values():
                total_ahd.extend(l)
            doc_ahd = np.mean(total_ahd)

            total_hd = list()
            for l in doc_hd.values():
                total_hd.extend(l)
            doc_hd = np.mean(total_hd)

            total_hd95 = list()
            for l in doc_hd95.values():
                total_hd95.extend(l)
            doc_hd95 = np.mean(total_hd95)

            total_iou = list()
            for l in doc_iou.values():
                total_iou.extend(l)
            doc_iou = np.mean(total_iou)

            total_acc = list()
            for l in doc_acc.values():
                total_acc.extend(l)
            doc_acc = np.mean(total_acc)

            self.doc_wise[input['file_name']] = {
                "AHD": doc_ahd,
                "IOU": doc_iou,
                "HD": doc_hd,
                "HD95": doc_hd95,
                "ACC": doc_acc
            }

    def evaluate(self):
        # save self.count somewhere, or print it, or return it.
        total_ahd = list()
        for l in self.ahd.values():
            total_ahd.extend(l)
        self.ahd["Overall"] = np.mean(total_ahd)

        total_hd = list()
        for l in self.hd.values():
            total_hd.extend(l)
        self.hd["Overall"] = np.mean(total_hd)

        total_hd95 = list()
        for l in self.hd95.values():
            total_hd95.extend(l)
        self.hd95["Overall"] = np.mean(total_hd95)

        total_iou = list()
        for l in self.iou.values():
            total_iou.extend(l)
        self.iou["Overall"] = np.mean(total_iou)

        total_acc = list()
        for l in self.acc.values():
            total_acc.extend(l)
        self.acc["Overall"] = np.mean(total_acc)

        import csv
        with open("metrics_per_doc.csv", 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["Image", "AHD", "IOU", "HD", "HD95", "ACC"])
            writer.writeheader()
            for filename, metrics in self.doc_wise.items():
                writer.writerow({"Image": filename, **metrics})

        """Tabular results"""
        # table = {
        #     "HD": self.hd["Overall"],
        #     "Avg HD": self.ahd["Overall"],
        #     "HD 95": self.hd95["Overall"],
        #     "IoU": 100 * self.iou["Overall"],
        # }
        #
        # table = create_small_table(table)
        # self._logger.info("\n", table)

        for each_region in categories_list:
            if len(self.hd[each_region]):
                self.hd[each_region] = np.mean(self.hd[each_region])
                self.ahd[each_region] = np.mean(self.ahd[each_region])
                self.hd95[each_region] = np.mean(self.hd95[each_region])
                self.iou[each_region] = np.mean(self.iou[each_region])
                self.acc[each_region] = np.mean(self.acc[each_region])
            else:
                self.hd[each_region] = -1
                self.ahd[each_region] = -1
                self.hd95[each_region] = -1
                self.iou[each_region] = -1
                self.acc[each_region] = -1

        """Utils for tabular results"""
        # table_hd = [item for item in self.hd.items()]
        # table_ahd = [item for item in self.ahd.items()]
        # table_hd95 = [item for item in self.hd95.items()]
        # table_iou = [item for item in self.iou.items()]
        #
        # def make_table(l):
        #     # tabulate it
        #     N_COLS = min(6, len(l) * 2)
        #     results_flatten = list(itertools.chain(*l))
        #     results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        #     table = tabulate(
        #         results_2d,
        #         tablefmt="pipe",
        #         floatfmt=".3f",
        #         headers=["category", "AP"] * (N_COLS // 2),
        #         numalign="left",
        #     )
        #     self._logger.info("\n" + table)
        #
        # make_table(table_hd)
        # make_table(table_ahd)
        # make_table(table_hd95)
        # make_table(table_iou)

        # print("dv")
        results = {
            'count': {'total': self.count},
            "HD": self.hd,
            "Avg HD": self.ahd,
            "HD95": self.hd95,
            "IoU": self.iou,
            "Accuracy": self.acc,
        }

        return collections.OrderedDict(sorted(results.items()))
