import argparse
from typing import List

import torch
import onnx

import numpy as np

from lightglue_onnx import DISK, LightGlue, SIFT, LightGlueEnd2End, SuperPoint
from lightglue_onnx.end2end import normalize_keypoints
from lightglue_onnx.utils import load_image, rgb_to_grayscale
from lightglue_onnx.ops.sdpa import register_aten_sdpa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_size",
        nargs="+",
        type=int,
        default=512,
        required=False,
        help="Sample image size for ONNX tracing. If a single integer is given, resize the longer side of the image to this value. Otherwise, please provide two integers (height width).",
    )
    parser.add_argument(
        "--extractor_type",
        type=str,
        default="superpoint",
        choices=["superpoint", "disk", "sift"],
        required=False,
        help="Type of feature extractor. Supported extractors are 'superpoint' and 'disk'. Defaults to 'superpoint'.",
    )
    parser.add_argument(
        "--extractor_path",
        type=str,
        default=None,
        required=False,
        help="Path to save the feature extractor ONNX model.",
    )
    parser.add_argument(
        "--lightglue_path",
        type=str,
        default=None,
        required=False,
        help="Path to save the LightGlue ONNX model.",
    )
    parser.add_argument(
        "--end2end",
        action="store_true",
        help="Whether to export an end-to-end pipeline instead of individual models.",
    )
    parser.add_argument(
        "--dynamic", action="store_true", help="Whether to allow dynamic image sizes."
    )
    parser.add_argument(
        "--img_path",
        type=str,
        default=None,
        required=False,
        help="Path to load images",
    )

    # Extractor-specific args:
    parser.add_argument(
        "--max_num_keypoints",
        type=int,
        default=2048,
        required=False,
        help="Maximum number of keypoints outputted by the extractor.",
    )

    return parser.parse_args()


def export_onnx(
    img_size=512,
    extractor_type="superpoint",
    extractor_path=None,
    lightglue_path=None,
    img_path="/srv/calibrations/GlobalFootball2/01_11v11_soccer_amfb_green/",
    end2end=False,
    dynamic=False,
    max_num_keypoints=None,
):
    # Handle args
    if isinstance(img_size, List) and len(img_size) == 1:
        img_size = img_size[0]

    if extractor_path is not None and end2end:
        raise ValueError(
            "Extractor will be combined with LightGlue when exporting end-to-end model."
        )
    if extractor_path is None:
        extractor_path = f"weights/{extractor_type}.onnx"
        if max_num_keypoints is not None:
            extractor_path = extractor_path.replace(
                ".onnx", f"_{max_num_keypoints}.onnx"
            )

    if lightglue_path is None:
        lightglue_path = (
            f"weights/{extractor_type}_lightglue"
            f"{'_end2end' if end2end else ''}"
            ".onnx"
        )

    # Sample images for tracing
    clip_percentage = 0.10
    image0_clip_coord = int(4104 * (1-clip_percentage))
    image1_clip_coord = int(4104 * clip_percentage)

    # Image size: 410 x 3046
    image0 = load_image(img_path + "/image0-synchronized.jpg")[:, :, image0_clip_coord:]
    image1 = load_image(img_path + "/image1-synchronized.jpg")[:, :, :image1_clip_coord]

    # Models
    extractor_type = extractor_type.lower()
    if extractor_type == "superpoint":
        # SuperPoint works on grayscale images.
        image0 = rgb_to_grayscale(image0)
        image1 = rgb_to_grayscale(image1)
        extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval()
        lightglue = LightGlue(extractor_type).eval()
    elif extractor_type == "disk":
        extractor = DISK(max_num_keypoints=max_num_keypoints).eval()
        lightglue = LightGlue(extractor_type).eval()
    elif extractor_type == "sift":
        extractor = SIFT(max_num_keypoints=max_num_keypoints).eval()
        lightglue = LightGlue(extractor_type, filter_threshold=0.98, depth_confidence=-1, width_confidence=-1).eval()        
    else:
        raise NotImplementedError(
            f"LightGlue has not been trained on {extractor_type} features."
        )

    # ONNX Export

    # Export Extractor
    dynamic_axes = {
        "keypoints": {1: "num_keypoints"},
        "scores": {1: "num_keypoints"},
        "descriptors": {1: "num_keypoints"},
    }
    if dynamic:
        dynamic_axes.update({"image": {2: "height", 3: "width"}})
    else:
        print(
            f"WARNING: Exporting without --dynamic implies that the {extractor_type} extractor's input image size will be locked to {image0.shape[-2:]}"
        )
        extractor_path = extractor_path.replace(
            ".onnx", f"_{image0.shape[-2]}x{image0.shape[-1]}.onnx"
        )

    # torch.onnx.export(
    #     extractor,
    #     image0[None],
    #     extractor_path,
    #     input_names=["image"],
    #     output_names=["keypoints", "scores", "descriptors"],
    #     opset_version=17,
    #     dynamic_axes=dynamic_axes,
    # )

    # Export LightGlue
    feats0, feats1 = extractor.extract(image0), extractor.extract(image1)

    # Keypoints: 410 x 3046
    kpts0, desc0 = feats0["keypoints"], feats0["descriptors"]
    kpts1, desc1 = feats1["keypoints"], feats1["descriptors"]

    kpts0 = normalize_keypoints(kpts0, image0.shape[1], image0.shape[2])
    kpts1 = normalize_keypoints(kpts1, image1.shape[1], image1.shape[2])

    kpts0 = torch.cat(
        [kpts0] + [feats0[k].unsqueeze(-1) for k in ("scales", "oris")], -1
    )
    kpts1 = torch.cat(
        [kpts1] + [feats1[k].unsqueeze(-1) for k in ("scales", "oris")], -1
    )

    print(f"Found {kpts0.shape[1]} keypoints in image 0")
    print(f"Found {kpts1.shape[1]} keypoints in image 1")

    register_aten_sdpa()

    def pad_kpts(tensor, target_size):
        x = torch.ones(target_size - tensor.shape[1], 1) * -1
        y = torch.ones(target_size - tensor.shape[1], 1) * -1

        padding = torch.cat(
            [x] + [y] + [torch.zeros(target_size - tensor.shape[1], 2)], -1
        )

        # minimum = torch.min(tensor, dim=1).values
        # padding = minimum.repeat(target_size - tensor.shape[1], 1)
        return torch.cat((tensor[0], padding))[None]

    def pad_desc(tensor, target_size):
        padding = torch.zeros(target_size - tensor.shape[1], tensor.shape[2])
        return torch.cat((tensor[0], padding))[None]

    # Add padding to kpts0 to get total keypoints
    # print(max_num_keypoints)
    # kpts0 = pad_kpts(kpts0, max_num_keypoints)
    # kpts1 = pad_kpts(kpts1, max_num_keypoints)
    # desc0 = pad_desc(desc0, max_num_keypoints)
    # desc1 = pad_desc(desc1, max_num_keypoints)

    # Export as numpy arrays so they can be loaded in native-camera-tools
    np.save('/srv/calibrations/GlobalFootball/01_11v11_soccer_amfb_green/kpts0.npy', kpts0.detach().cpu().numpy())
    np.save('/srv/calibrations/GlobalFootball/01_11v11_soccer_amfb_green/desc0.npy', desc0.detach().cpu().numpy())
    np.save('/srv/calibrations/GlobalFootball/01_11v11_soccer_amfb_green/kpts1.npy', kpts1.detach().cpu().numpy())
    np.save('/srv/calibrations/GlobalFootball/01_11v11_soccer_amfb_green/desc1.npy', desc1.detach().cpu().numpy())
    print("saved")

    print(kpts0)
    print(kpts1)

    torch.onnx.export(
        lightglue,
        (kpts0, kpts1, desc0, desc1),
        lightglue_path,
        input_names=["kpts0", "kpts1", "desc0", "desc1"],
        output_names=["scores"],
        opset_version=11,
        dynamic_axes={
            "kpts0": {1: "num_keypoints0"},
            "kpts1": {1: "num_keypoints1"},
            "desc0": {1: "num_keypoints0"},
            "desc1": {1: "num_keypoints1"},
            #"scores": {0: "num_matches0", 1: "num_matches1"},
        },
    )

    # Fix bug in Flatten node
    # def fix_flatten(model):
    #     for node in model.graph.node:
    #         if node.name == "/Flatten":
    #             print(node.attribute)
    #             node.attribute[0].i = 0
    #             print(node.attribute)

    # print ("Fixing model..")
    # model = onnx.load(lightglue_path)
    # new_filename = lightglue_path.replace(".onnx", "_fixed.onnx")
    # fix_flatten(model)
    # onnx.save(model, new_filename)

    # from lightglue_onnx.lightglue import TestModel
    # from lightglue_onnx.lightglue import filter_matches

    # model = TestModel()
    # scores = torch.randn([1, 736, 858])
    # torch.onnx.export(model, (scores), 'weights/filter_matches.onnx', 
    #                 input_names=["scores"],
    #                 output_names=["matches0", "mscores0"], 
    #                 # dynamic_axes={
    #                 #     "matches0": {0: "num_matches0"},
    #                 #     "mscores0": {0: "num_matches0"},
    #                 # },
    #                 opset_version=11)

    # print ("Fixing model..")
    # model = onnx.load("weights/filter_matches.onnx")
    # new_filename = "weights/filter_matches.onnx".replace(".onnx", "_fixed.onnx")
    # fix_flatten(model)
    # onnx.save(model, new_filename)


if __name__ == "__main__":
    args = parse_args()
    export_onnx(**vars(args))
