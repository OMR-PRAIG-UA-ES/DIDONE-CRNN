import os
import cv2
import json
import torch
from types import SimpleNamespace
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2
from torchvision.tv_tensors import Image as v2Image
from skimage.filters import threshold_sauvola


class ResizeByHeight:
    def __init__(self, height):
        self.height = height

    def __call__(self, img):
        _, h, w = img.shape
        new_w = int(w * self.height / h)
        return v2.functional.resize(img, (self.height, new_w))


class SauvolaThreshold:
    def __init__(self, window_size=15, k=0.5, r=128):
        self.window_size = window_size
        self.k = k
        self.r = r

    def __call__(self, image):
        image = image.cpu().numpy()
        mask = threshold_sauvola(
            image, window_size=self.window_size, k=self.k, r=self.r
        )
        return v2Image((image > mask) * 255)


class Normalize:
    def __call__(self, img):
        return (255 - img) / 255


class MusicSymbolDataset(Dataset):
    def __init__(self, root_path: str, encoding: str, img_height: int):
        root_path = root_path[:-1] if root_path.endswith("/") else root_path
        self.root_path = root_path
        assert encoding in [
            "std",
            "split",
        ], "Invalid encoding, must be either 'std' or 'split'"
        self.encoding = encoding
        self.__training = True
        self.__img_height = img_height
        self.augments = self.__load_augmentations()

    def __load_augmentations(self):
        return v2.Compose(
            [
                v2.ToImage(),
                ResizeByHeight(self.img_height),
                v2.RandomApply(
                    [v2.ElasticTransform(alpha=0.25)], p=0.2 if self.training else 0
                ),
                v2.RandomApply(
                    [
                        v2.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.2 if self.training else 0,
                ),
                v2.RandomApply(
                    [v2.GaussianBlur(kernel_size=23)], p=0.2 if self.training else 0
                ),
                v2.Grayscale(),
                SauvolaThreshold(25, 0.2),
                Normalize(),
                v2.RandomRotation(degrees=3),
            ]
        )

    @property
    def img_height(self):
        return self.__img_height

    @img_height.setter
    def img_height(self, value):
        self.__img_height = value
        self.augments = self.__load_augmentations()

    @property
    def training(self):
        return self.__training

    @training.setter
    def training(self, value):
        self.__training = value
        self.augments = self.__load_augmentations()


class DidoneDataset(MusicSymbolDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        system_path = os.path.join(self.root_path, "system")
        files = [
            os.path.join(system_path, file.split(".")[0])
            for file in os.listdir(system_path)
        ]
        self.files = list(set(files))
        with open(os.path.join(self.root_path, "dicts", f"{self.encoding}.json"), "r") as f:
            c2i = json.load(f)
        self.__vocab = SimpleNamespace(c2i=c2i, i2c={v: k for k, v in c2i.items()})

    @property
    def vocab(self):
        return self.__vocab

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = cv2.imread(f"{self.files[idx]}.jpg")
        img = self.augments(img)
        with open(f"{self.files[idx]}.{self.encoding}.txt", "r") as f:
            target = [int(i) for i in f.read().split(" ")]

        return img, torch.tensor(target, dtype=torch.int32)


class PrimusDataset(MusicSymbolDataset):
    def __init__(self, camera: bool, **kwargs):
        super().__init__(**kwargs)
        self.files = [
            os.path.join(self.root_path, folder)
            for folder in os.listdir(self.root_path)
        ]
        self.camera = camera

        parent = os.path.dirname(self.root_path)
        if not os.path.exists(f"{parent}/dicts/{self.encoding}.json"):
            self.__vocab = self.__generate_vocab(parent)
        else:
            with open(os.path.join(parent, "dicts", f"{self.encoding}.json"), "r") as f:
                c2i = json.load(f)
            self.__vocab = SimpleNamespace(c2i=c2i, i2c={v: k for k, v in c2i.items()})

    def __unique_symbols_from_file(self, file):
        with open(file, "r") as f:
            data = json.load(f)

        symbols = set()
        for page in data["pages"]:
            for region in page["regions"]:
                if region["type"] == "staff" and "symbols" in region:
                    for symbol in region["symbols"]:
                        if self.encoding == "split":
                            symbols.add(symbol["agnostic_symbol_type"])
                            symbols.add(symbol["position_in_staff"])
                        else:
                            symbols.add(
                                f"{symbol['agnostic_symbol_type']}:{symbol['position_in_staff']}"
                            )
        return symbols

    def __generate_vocab(self, directory):
        c2i = {}
        for file in self.files:
            basename = os.path.basename(file)
            symbols = self.__unique_symbols_from_file(
                os.path.join(file, f"{basename}.json")
            )
            for symbol in symbols:
                if symbol not in c2i:
                    c2i[symbol] = len(c2i)
        c2i["<BLANK>"] = len(c2i)
        c2i["<PAD>"] = len(c2i)

        os.makedirs(f"{directory}/dicts", exist_ok=True)
        with open(f"{directory}/dicts/{self.encoding}.json", "w") as f:
            json.dump(c2i, f)
        return SimpleNamespace(c2i=c2i, i2c={v: k for k, v in c2i.items()})

    @property
    def vocab(self):
        return self.__vocab

    def __len__(self):
        return len(self.files)

    def __get_img_from_idx(self, idx: int):
        filename = "distorted.jpg" if self.camera else "kern.png"
        img = cv2.imread(f"{self.files[idx]}/{filename}")
        if img is None:
            raise ValueError(f"{self.files[idx]}/{filename} not found")
        img = self.augments(img)
        return img

    def __get_seq_from_idx(self, idx: int):
        basename = os.path.basename(self.files[idx])
        with open(f"{self.files[idx]}/{basename}.json", "r") as f:
            data = json.load(f)

        target = []
        for page in data["pages"]:
            for region in page["regions"]:
                if region["type"] == "staff" and "symbols" in region:
                    for symbol in region["symbols"]:
                        if self.encoding == "split":
                            target.extend(
                                [
                                    self.vocab.c2i[symbol["agnostic_symbol_type"]],
                                    self.vocab.c2i[symbol["position_in_staff"]],
                                ]
                            )
                        else:
                            target.append(
                                self.vocab.c2i[
                                    f"{symbol['agnostic_symbol_type']}:{symbol['position_in_staff']}"
                                ]
                            )
        if len(target) == 0:
            raise ValueError(f"No symbols found in folder {self.files[idx]}")
        return target

    def __getitem__(self, idx):
        x = self.__get_img_from_idx(idx)
        y = self.__get_seq_from_idx(idx)
        y = torch.tensor(y, dtype=torch.int32)
        return x, y
