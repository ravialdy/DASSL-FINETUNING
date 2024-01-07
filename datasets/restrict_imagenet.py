import os
import pickle
from collections import OrderedDict

from my_dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from my_dassl.utils import listdir_nohidden, mkdir_if_missing

from .imagenet import ImageNet
from .oxford_pets import OxfordPets

@DATASET_REGISTRY.register()
class RestrictedImageNet(ImageNet):

    dataset_dir = "imagenet"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        self.cfg = cfg
        self.restricted_labels, self.label_mapping = self.define_restricted_labels()
        mkdir_if_missing(self.split_fewshot_dir)

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = self.read_classnames(text_file)
        train = self.read_data(classnames, "train")

        preprocessed = {"train": train, "test": test}
        with open(self.preprocessed, "wb") as f:
            pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train = data["train"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                data = {"train": train}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, test = OxfordPets.subsample_classes(train, test, subsample=subsample)

        super().__init__(train_x=train, val=test, test=test)

    def define_restricted_labels(self):
        restricted_ranges = {
            'Dog': (151, 268),
            'Cat': (281, 285),
            'Frog': (30, 32),
            'Turtle': (33, 37),
            'Bird': (80, 100),
            'Monkey': (365, 382),
            'Fish': (389, 397),
            'Crab': (118, 121),
            'Insect': (300, 319)
        }

        restricted_labels = {}
        label_mapping = {}
        new_idx = 0
        for label, (start, end) in restricted_ranges.items():
            for idx in range(start, end + 1):
                restricted_labels[idx] = label
                label_mapping[idx] = new_idx
                new_idx += 1

        return restricted_labels, label_mapping
    
    @staticmethod
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames

    def read_data(self, classnames, split_dir):
        split_dir = os.path.join(self.image_dir, split_dir)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        items = []

        for label, folder in enumerate(folders):
            if label not in self.restricted_labels:
                continue

            new_label = self.label_mapping[label]
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                item = Datum(impath=impath, label=new_label, classname=classname)
                items.append(item)

        return items