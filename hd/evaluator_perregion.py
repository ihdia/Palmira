"""
HD per region
Useful for getting instance level metrics for comparison
"""
import collections
import os.path as osp

import cv2
import numpy as np
from detectron2.data import DatasetCatalog
from detectron2.evaluation import DatasetEvaluator
from medpy.metric import hd95, assd
from scipy.interpolate import splprep, splev

from indiscapes_dataset import categories_list


def _proc_annotations(annotations):
    dic = {}
    for file in annotations.copy():
        for region in file["annotations"]:
            l = region["segmentation"][0]
            n = 2
            x = [l[i: i + n] for i in range(0, len(l), n)]
            region["segmentation"] = x
        dic[file["file_name"]] = file

        segm = dic[file["file_name"]]["annotations"]
        segm_per_region = {i: [] for i in range(len(categories_list))}
        for region in segm:
            segm_per_region[region["category_id"]].append(region["segmentation"])
        dic[file["file_name"]]["segm_per_region"] = segm_per_region
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
    predmasks = output["instances"].pred_masks
    predmasks = (predmasks.cpu()) * 255
    predmasks = np.uint8(predmasks.numpy())

    segm_per_region = {i: [] for i in range(len(categories_list))}

    for i in range(len(output["instances"])):
        contours, hierarchy = cv2.findContours(predmasks[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        pred_class = categories_list[output["instances"].pred_classes[i]]
        if pred_class == "Character Line Segment":
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
        segm_per_region[output["instances"].pred_classes[i].item()].append(smoothened)
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

        self.metrics_for_csv = []
        # self.write_to_csv = True  # If true, only csv. No images
        self.write_to_csv = False  # If true, only csv. No images

    def reset(self):
        self.count = 0
        self.ahd = {cat: [] for cat in categories_list}
        self.hd = {cat: [] for cat in categories_list}
        self.hd95 = {cat: [] for cat in categories_list}
        self.iou = {cat: [] for cat in categories_list}
        self.acc = {cat: [] for cat in categories_list}
        self.doc_wise = {}

        self.metrics_for_csv = []
        # self.write_to_csv = True  # If true, only csv. No images
        self.write_to_csv = False  # If true, only csv. No images

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            self.count += len(output["instances"])
            gt_segm = self.annotations[input["file_name"]]["segm_per_region"]
            try:
                _ = output["instances"].pred_masks
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
                    """"Peformed using medpy"""
                    gt_mask = np.zeros((len(gt), input["height"], input["width"]), dtype=np.int8)
                    for i in range(len(gt)):
                        cv2.fillPoly(gt_mask[i], np.array([gt[i]]).astype(np.int32), 1)
                    pred_mask = np.zeros((len(pred), input["height"], input["width"]), dtype=np.int8)
                    for i in range(len(pred)):
                        cv2.fillPoly(pred_mask[i], np.array([pred[i]]).astype(np.int32), 1)
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

                    iou_hd_dict = np.empty((len(gt), len(pred), 3))
                    # IOU, HD95, AHD
                    for i, each_gt_instance in enumerate(gt_mask):
                        for j, each_pred_instance in enumerate(pred_mask):
                            inst_iou, _ = compute_iou_and_accuracy(gt_mask[i], pred_mask[j])
                            inst_hd95 = hd95(gt_mask[i], pred_mask[j])
                            inst_ahd = assd(gt_mask[i], pred_mask[j])
                            iou_hd_dict[i][j][0] = inst_iou
                            iou_hd_dict[i][j][1] = inst_hd95
                            iou_hd_dict[i][j][2] = inst_ahd
                    corr_matrix = np.empty((len(gt), 4))
                    corr_matrix[:, 0] = np.argmax(iou_hd_dict[:, :, 0], 1)
                    for i, each_gt_instance_metric in enumerate(iou_hd_dict):
                        corr_matrix[i, 1:] = each_gt_instance_metric[corr_matrix[i, 0].astype(np.int)]

                    gt_mask_copy = gt_mask.copy()
                    gt_mask_copy = np.repeat(gt_mask_copy[:, :, :, np.newaxis], 3, axis=3)

                    def bbox2(img):
                        rows = np.any(img, axis=1)
                        cols = np.any(img, axis=0)
                        rmin, rmax = np.where(rows)[0][[0, -1]]
                        cmin, cmax = np.where(cols)[0][[0, -1]]

                        return rmin, rmax, cmin, cmax

                    for i, each_gt_instance in enumerate(gt_mask):
                        rmin, rmax, cmin, cmax = bbox2(each_gt_instance)
                        pos = (int(cmin), int((rmin + rmax) / 2))
                        gt_mask_copy[i] = cv2.cvtColor(each_gt_instance, cv2.COLOR_GRAY2RGB)
                        cv2.putText(gt_mask_copy[i], f"{i}", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)

                    gt_image = gt_mask_copy.sum(axis=0).clip(0, 255).astype(np.uint8)
                    # x = plt.imshow(gt_image)
                    # plt.show()
                    pred_image = pred_mask.sum(axis=0).clip(0, 255)
                    pred_image_copy = pred_image.copy()
                    pred_image_copy = np.repeat(pred_image_copy[:, :, np.newaxis], 3, axis=2).astype(np.uint8)

                    for i, each_pred_instance in enumerate(pred_mask):
                        rmin, rmax, cmin, cmax = bbox2(each_pred_instance)
                        pos = (int(cmin), int((rmin + rmax) / 2))
                        # pred_image_copy[i] = cv2.cvtColor(each_pred_instance, cv2.COLOR_GRAY2RGB)
                        if i in corr_matrix[:, 0].astype(np.int):
                            if self.write_to_csv:
                                metrics = (corr_matrix[np.where(i == corr_matrix[:, 0]), 1:]).squeeze()
                                if metrics.ndim > 1:
                                    metrics = metrics[np.argmax(metrics[:, 0])]
                                gt_idx = np.where((metrics == corr_matrix[:, 1:]).all(axis=1))[0].item()
                                self.metrics_for_csv.append(
                                    {
                                        "region_name": f"{osp.splitext('_'.join(input['file_name'].split('/')[-3:]))[0]}_gt_{reg_type}_{gt_idx}",
                                        "iou": metrics[0].round(2),
                                        "ahd": metrics[2].round(2),
                                        "hd95": metrics[1].round(2),
                                    }
                                )
                                continue
                            text_metrics = np.array2string(
                                (corr_matrix[np.where(i == corr_matrix[:, 0]), 1:].round(4)).squeeze()
                            )
                            cv2.putText(
                                pred_image_copy,
                                f"{i} - {text_metrics}",
                                pos,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (255, 0, 0),
                                2,
                            )
                        else:
                            cv2.putText(pred_image_copy, f"{i}", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                    # x = plt.imshow(pred_image_copy)
                    # plt.show()
                    if not self.write_to_csv:
                        save_path = "final_outputs/comparision/bmrcnn_masks_whew/"
                        import os

                        file_name = os.path.splitext("_".join(input["file_name"].split("/")[-3:]))[0]
                        cv2.imwrite(
                            save_path + f"{file_name}_gt_{reg_type}.jpg", cv2.cvtColor(gt_image, cv2.COLOR_RGB2BGR)
                        )
                        cv2.imwrite(
                            save_path + f"{file_name}_pred_{reg_type}.jpg",
                            cv2.cvtColor(pred_image_copy, cv2.COLOR_RGB2BGR),
                        )
                    # res_iou, res_accuracy = compute_iou_and_accuracy(pred_mask, gt_mask)
                    # res_ahd, res_hd, res_hd95 = assd(pred_mask, gt_mask), hd(pred_mask, gt_mask), hd95(pred_mask, gt_mask)
                    # self.ahd[categories_list[reg_type]].append(res_ahd)
                    # self.hd[categories_list[reg_type]].append(res_hd)
                    # self.hd95[categories_list[reg_type]].append(res_hd95)
                    # self.hd[categories_list[reg_type]].append(hd(pred_mask, gt_mask))
                    # self.iou[categories_list[reg_type]].append(res_iou)
                    # self.acc[categories_list[reg_type]].append(res_accuracy)
                    #
                    # doc_ahd[categories_list[reg_type]].append(res_ahd)
                    # doc_hd[categories_list[reg_type]].append(res_hd)
                    # doc_hd95[categories_list[reg_type]].append(res_hd95)
                    # doc_hd[categories_list[reg_type]].append(hd(pred_mask, gt_mask))
                    # doc_iou[categories_list[reg_type]].append(res_iou)
                    # doc_acc[categories_list[reg_type]].append(res_accuracy)
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

            self.doc_wise[input["file_name"]] = {
                "AHD": doc_ahd,
                "IOU": doc_iou,
                "HD": doc_hd,
                "HD95": doc_hd95,
                "ACC": doc_acc,
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

        """Exporting to file"""
        # with open("metrics_per_doc.csv", 'w') as csvfile:
        #     writer = csv.DictWriter(csvfile, fieldnames=["Image", "AHD", "IOU", "HD", "HD95", "ACC"])
        #     writer.writeheader()
        #     for filename, metrics in self.doc_wise.items():
        #         writer.writerow({"Image":filename, **metrics})

        with open("region_wise.csv", "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["region_name", "iou", "ahd", "hd95"])
            writer.writeheader()
            for metrics in self.metrics_for_csv:
                writer.writerow(metrics)

        """Printing as a detectron2 table"""
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

        """Some utils for detectron2 table"""
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

        results = {
            "count": {"total": self.count},
            "HD": self.hd,
            "Avg HD": self.ahd,
            "HD95": self.hd95,
            "IoU": self.iou,
            "Accuracy": self.acc,
        }

        return collections.OrderedDict(sorted(results.items()))
