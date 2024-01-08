import os
import pickle
from collections import OrderedDict

from my_dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from my_dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class ImageNet(DatasetBase):

    dataset_dir = "imagenet"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        self.cfg = cfg
        mkdir_if_missing(self.split_fewshot_dir)

        self.text_file = os.path.join(self.dataset_dir, "classnames.txt")
        self.classes_attacked = set()  # Track classes that have been attacked
        classnames = self.read_classnames(self.text_file)
        train = self.read_data(classnames, "train")


        if cfg.TEST.ADVERSARIAL_FOLDER: test = self.read_data_adv(classnames, f"Adv_Sample_{cfg.TEST.SAMPLE}", cfg.TEST.ADVERSARIAL_ATTACK, cfg.TEST.REVERSE)        
        elif cfg.TEST.SAMPLE.isdigit(): test = self.read_data(classnames, f"Sample_{cfg.TEST.SAMPLE}")
        else: test = self.read_data(classnames, "val")

        preprocessed = {"train": train, "test": test}
        with open(self.preprocessed, "wb") as f:
            pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        # print("Debugging train paths before few-shot processing:")
        # print([item.impath for item in train[:5]])

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
    
    def read_data_adv(self, classnames, split_dir, attack, reverse):
        if reverse == "": 
            if self.cfg.MODEL.TORCHVISION == "": split_dir = os.path.join(self.image_dir, split_dir, self.cfg.TRAINER.NAME, attack)
            else: split_dir = os.path.join(self.image_dir, split_dir, self.cfg.MODEL.TORCHVISION, attack)
        elif reverse == "TRAINER": split_dir = os.path.join(self.image_dir, split_dir, self.cfg.MODEL.TORCHVISION, attack)
        elif reverse == "TORCHVISION": split_dir = os.path.join(self.image_dir, split_dir, self.cfg.TRAINER.NAME, attack)
        print(f"Going to split_dir: {split_dir}")
        with open(self.text_file, 'r') as file: classnames = [line.strip().split(' ')[0] for line in file.readlines()]
        imnames = listdir_nohidden(split_dir)
        items = []
        for imname in imnames:
            label = int(imname.split("_")[-1].replace(".png", ""))  # Extract label from filename
            impath = os.path.join(split_dir, imname)
            classname = classnames[label]  # Get classname using label
            item = Datum(impath=impath, label=label, classname=classname)
            items.append(item)
        return items

    def read_data(self, classnames, split_dir):
        # print(f"Debugging paths:")
        # print(f"self.image_dir in read_data: {self.image_dir}")
        # print(f"split_dir parameter in read_data: {split_dir}")
        split_dir = os.path.join(self.image_dir, split_dir)
        print(f"split_dir in read_data: {split_dir}")

        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        # print(f"folders in read_data: {split_dir}")
        items = []

        for label, folder in enumerate(folders):
            # Skip the classes that are less than cfg.TEST.START_CLASS
            if label < self.cfg.TEST.START_CLASS:
                continue
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items
