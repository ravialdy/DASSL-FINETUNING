import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, confusion_matrix

from .build import EVALUATOR_REGISTRY


class EvaluatorBase:
    """Base evaluator."""

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


@EVALUATOR_REGISTRY.register()
class Classification(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self._y_true = []
        self._y_pred = []
        self._y_pred_att = []
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        self._y_pred_att = []
        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)
        self._radius_list = []

    def process(self, mo, gt, radius=None):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        if len(mo.shape) == 1:
            pred = mo
        else:
            pred = mo.max(1)[1]
        matches = pred.eq(gt).float()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())

        if self._per_class_res is not None:
            for i, label in enumerate(gt):
                label = label.item()
                matches_i = int(matches[i].item())
                self._per_class_res[label].append(matches_i)

        if radius is not None:
            self._radius_list.append(radius)

    def process_perturbed(self, mo_att):
        if len(mo_att.shape) == 1:
            pred_att = mo_att
        else:
            pred_att = mo_att.max(1)[1]

        self._y_pred_att.extend(pred_att.data.cpu().numpy().tolist())
        # import pdb; pdb.set_trace()

    def evaluate_att(self, y_true, y_pred_att):
        assert len(y_true) == len(y_pred_att), "Mismatch in ground truth and prediction lengths"
        results = OrderedDict()
        # Assuming self._y_true contains ground truth labels and self._y_pred_att contains labels predicted from perturbed data
        acc = 100.0 * np.sum(np.array(y_true) == np.array(y_pred_att)) / len(y_true)
        err = 100.0 - acc
        macro_f1 = 100.0 * f1_score(
            y_true,
            y_pred_att,
            average="macro",
            labels=np.unique(y_true)
        )

        # The first value will be returned by trainer.test_attack()
        results["accuracy"] = acc
        results["error_rate"] = err
        results["macro_f1"] = macro_f1

        print(
            "=> result\n"
            f"* total: {len(y_true):,}\n"
            f"* correct: {np.sum(y_true == y_pred_att):,}\n"
            f"* accuracy: {acc:.1f}%\n"
            f"* error: {err:.1f}%\n"
            f"* macro_f1: {macro_f1:.1f}%"
        )

        return results

    def evaluate(self):
        results = OrderedDict()
        acc = 100.0 * self._correct / self._total
        err = 100.0 - acc
        macro_f1 = 100.0 * f1_score(
            self._y_true,
            self._y_pred,
            average="macro",
            labels=np.unique(self._y_true)
        )

        # The first value will be returned by trainer.test()
        results["accuracy"] = acc
        results["error_rate"] = err
        results["macro_f1"] = macro_f1

        print(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* correct: {self._correct:,}\n"
            f"* accuracy: {acc:.1f}%\n"
            f"* error: {err:.1f}%\n"
            f"* macro_f1: {macro_f1:.1f}%"
        )

        if self._per_class_res is not None:
            labels = list(self._per_class_res.keys())
            labels.sort()

            print("=> per-class result")
            accs = []

            for label in labels:
                classname = self._lab2cname[label]
                res = self._per_class_res[label]
                correct = sum(res)
                total = len(res)
                acc = 100.0 * correct / total
                accs.append(acc)
                print(
                    f"* class: {label} ({classname})\t"
                    f"total: {total:,}\t"
                    f"correct: {correct:,}\t"
                    f"acc: {acc:.1f}%"
                )
            mean_acc = np.mean(accs)
            print(f"* average: {mean_acc:.1f}%")

            results["perclass_accuracy"] = mean_acc

        if self.cfg.TEST.COMPUTE_CMAT:
            cmat = confusion_matrix(
                self._y_true, self._y_pred, normalize="true"
            )
            save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")
            torch.save(cmat, save_path)
            print(f"Confusion matrix is saved to {save_path}")

        if self._radius_list:
            avg_radius = np.mean(self._radius_list)
            print(f"* average_radius: {avg_radius:.3f}")
            results["average_radius"] = avg_radius

        return results