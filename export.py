import argparse
from typing import List

import torch

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

    # Extractor-specific args:
    parser.add_argument(
        "--max_num_keypoints",
        type=int,
        default=None,
        required=False,
        help="Maximum number of keypoints outputted by the extractor.",
    )

    return parser.parse_args()


def export_onnx(
    img_size=512,
    extractor_type="superpoint",
    extractor_path=None,
    lightglue_path=None,
    img0_path="assets/sacre_coeur1.jpg",
    img1_path="assets/sacre_coeur2.jpg",
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
    image0, scales0 = load_image(img0_path, resize=img_size)
    image1, scales1 = load_image(img1_path, resize=img_size)
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
        lightglue = LightGlue(extractor_type).eval()        
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
    feats0, feats1 = extractor(image0[None]), extractor(image1[None])
    kpts0, scores0, desc0 = feats0
    kpts1, scores1, desc1 = feats1

    kpts0 = normalize_keypoints(kpts0, image0.shape[1], image0.shape[2])
    kpts1 = normalize_keypoints(kpts1, image1.shape[1], image1.shape[2])

    register_aten_sdpa()




    torch.onnx.export(
        lightglue,
        (kpts0, kpts1, desc0, desc1),
        lightglue_path,
        input_names=["kpts0", "kpts1", "desc0", "desc1"],
        output_names=["matches0", "mscores0"],
        opset_version=11,
        dynamic_axes={
            "kpts0": {1: "num_keypoints0"},
            "kpts1": {1: "num_keypoints1"},
            "desc0": {1: "num_keypoints0"},
            "desc1": {1: "num_keypoints1"},
            "matches0": {0: "num_matches0"},
            "mscores0": {0: "num_matches0"},
        },
    )
    
    # from lightglue_onnx.lightglue import TestModel
    #from lightglue_onnx.lightglue import filter_matches

    # model = TestModel()
    # scores = torch.randn([1, 736, 858])
    # torch.onnx.export(model, (scores), 'weights/filter_matches.onnx', 
    #                 input_names=["scores"],
    #                 output_names=["matches0", "mscores0"], 
    #                 opset_version=11)


if __name__ == "__main__":
    args = parse_args()
    export_onnx(**vars(args))
