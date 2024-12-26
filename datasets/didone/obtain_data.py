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
parser.add_argument("--only-dicts", action="store_true", help="Only generate dictionaries")
args = parser.parse_args()

os.makedirs("./data", exist_ok=True)
os.makedirs("./dicts", exist_ok=True)

def load_img_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    img_array = np.array(image)
    return img_array

split_symbols = {
    '<BLANK>': 0,
    '<PAD>': 1,
}
agnostic_symbols = {
    '<BLANK>': 0,
    '<PAD>': 1,
}

if args.only_dicts:
    print("Only generating dictionaries")

files = os.listdir("./files")
if args.limit:
    files = files[:args.limit]

images_in_already_in_data = set()
for file in os.listdir("./data"):
    if file.endswith(".jpg"):
        images_in_already_in_data.add("_".join(file.split("_")[:-1]))

for file in tqdm(files):
    path = os.path.join("./files", file)
    with open(path, "r") as f:
        data = json.load(f)

    for document in data["documents"]:
        identifier = document["name"].strip().replace(" ", "_")
        if not args.only_dicts and identifier in images_in_already_in_data:
            print(f"Skipping {identifier}")
            continue

        for section in document["sections"]:
            for image in section["images"]:
                e2e_data = []
                img_array = None
                for page in image["pages"]:
                    for region in page["regions"]:
                        if "bounding_box" not in region:
                            continue

                        region_bbox = [
                            region["bounding_box"].get(k)
                            for k in ("fromX", "fromY", "toX", "toY")
                        ]

                        if region["type"] == "staff" and "symbols" in region:
                            name = f"{identifier}_{region['id']}"
                            
                            if not args.only_dicts:
                                if img_array is None:
                                    img_array = load_img_from_url(image["url"])

                                region_crop = img_array[
                                    region_bbox[1] : region_bbox[3],
                                    region_bbox[0] : region_bbox[2],
                                ]

                                cv2.imwrite(
                                    f"./data/{name}.jpg",
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

                            with open(f"./data/{name}.std.txt", "w") as f:
                                f.write(" ".join(map(str, agnostic_sqnc)))

                            with open(f"./data/{name}.split.txt", "w") as f:
                                f.write(" ".join(map(str, split_sqnc)))

with open(f"./dicts/split.json", "w") as f:
    json.dump(split_symbols, f)

with open(f"./dicts/std.json", "w") as f:
    json.dump(agnostic_symbols, f)
