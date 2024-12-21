import json
import os
import argparse
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Obtain Didone samples")
parser.add_argument(
    "--limit", "-l",
    type=int,
    required=False,
    help="Limit the samples to load",
)
parser.add_argument("--img_height", "-ih", type=int, default=128, help="Image height")
args = parser.parse_args()

os.makedirs("./page", exist_ok=True)
os.makedirs("./system", exist_ok=True)
os.makedirs("./dicts", exist_ok=True)


def load_img_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    img_array = np.array(image)
    return img_array, image.width, image.height


def pascal2coco(voc):
    x_min, y_min, x_max, y_max = voc
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def pascal2yolo(voc, img_width, img_height):
    x_min, y_min, x_max, y_max = voc
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return [x_center, y_center, width, height]


split_symbols = {}
agnostic_symbols = {}

files = os.listdir("./files")
if args.limit:
    files = files[:args.limit]

for file in tqdm():
    path = os.path.join("./files", file)
    with open(path, "r") as f:
        data = json.load(f)

    for document in data["documents"]:
        identifier = document["name"].strip().replace(" ", "_")
        for section in document["sections"]:
            for image in section["images"]:
                img_array, img_width, img_height = load_img_from_url(image["url"])
                cv2.imwrite(f"./page/{identifier}.jpg", img_array)

                page_data = []
                e2e_data = []

                for page in image["pages"]:
                    page_bbox = [
                        page["bounding_box"].get(k)
                        for k in ("fromX", "fromY", "toX", "toY")
                    ]
                    page_coco = pascal2coco(page_bbox)
                    page_yolo = pascal2yolo(page_bbox, img_width, img_height)
                    page_data.append(
                        {
                            "voc": page_bbox,
                            "coco": page_coco,
                            "yolo": page_yolo,
                            "cls": "page",
                        }
                    )

                    for region in page["regions"]:
                        if "bounding_box" not in region:
                            continue

                        region_bbox = [
                            region["bounding_box"].get(k)
                            for k in ("fromX", "fromY", "toX", "toY")
                        ]
                        region_coco = pascal2coco(region_bbox)
                        region_yolo = pascal2yolo(region_bbox, img_width, img_height)
                        page_data.append(
                            {
                                "voc": region_bbox,
                                "coco": region_coco,
                                "yolo": region_yolo,
                                "cls": region["type"],
                            }
                        )

                        if region["type"] == "staff" and "symbols" in region:
                            region_crop = img_array[
                                region_bbox[1] : region_bbox[3],
                                region_bbox[0] : region_bbox[2],
                            ]

                            name = f"{identifier}_{region['id']}"
                            cv2.imwrite(
                                f"./system/{name}.jpg",
                                region_crop,
                            )

                            agnostic_sqnc = []
                            split_sqnc = []

                            for symbol in region["symbols"]:
                                if symbol["agnostic_symbol_type"] not in split_symbols:
                                    split_symbols[symbol["agnostic_symbol_type"]] = len(
                                        split_symbols
                                    )
                                if symbol["position_in_staff"] not in split_symbols:
                                    split_symbols[symbol["position_in_staff"]] = len(
                                        split_symbols
                                    )
                                split_sqnc.extend(
                                    [
                                        split_symbols[symbol["agnostic_symbol_type"]],
                                        split_symbols[symbol["position_in_staff"]],
                                    ]
                                )

                                # ---

                                agnostic = f"{symbol['agnostic_symbol_type']}:{symbol['position_in_staff']}"
                                if agnostic not in agnostic_symbols:
                                    agnostic_symbols[agnostic] = (
                                        len(agnostic_symbols)
                                    )

                                agnostic_sqnc.append(
                                    agnostic_symbols[agnostic]
                                )

                            with open(f"./system/{name}.std.txt", "w") as f:
                                f.write(" ".join(map(str, agnostic_sqnc)))

                            with open(f"./system/{name}.split.txt", "w") as f:
                                f.write(" ".join(map(str, split_sqnc)))

                with open(f"./page/{identifier}.json", "w") as f:
                    json.dump(page_data, f)

with open(f"./dicts/split.json", "w") as f:
    split_symbols["<BLANK>"] = len(split_symbols)
    split_symbols["<PAD>"] = len(split_symbols)
    json.dump(split_symbols, f)

with open(f"./dicts/std.json", "w") as f:
    agnostic_symbols["<BLANK>"] = len(agnostic_symbols)
    agnostic_symbols["<PAD>"] = len(agnostic_symbols)
    json.dump(agnostic_symbols, f)
