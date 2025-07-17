import json
from io import BytesIO

import litserve as ls
import numpy as np
from fastapi import Response, UploadFile
from PIL import Image
from pycocotools import mask as mask_utils

from lang_sam import LangSAM
from lang_sam.utils import draw_image

PORT = 8000

def rle_encode_binary_mask(mask: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


class BaseLangSAMAPI(ls.LitAPI):
    def setup(self, device: str) -> None:
        self.model = LangSAM(sam_type="sam2.1_hiera_small", device=device)
        print("LangSAM model initialized.")

    def decode_request(self, request) -> dict:
        sam_type = request.get("sam_type")
        box_threshold = float(request.get("box_threshold", 0.3))
        text_threshold = float(request.get("text_threshold", 0.25))
        text_prompt = request.get("text_prompt", "")
        image_file: UploadFile = request.get("image")
        if image_file is None:
            raise ValueError("No image file provided.")
        image_bytes = image_file.file.read()
        return {
            "sam_type": sam_type,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
            "image_bytes": image_bytes,
            "text_prompt": text_prompt,
        }

    def run_inference(self, inputs: dict) -> tuple[Image.Image, dict]:
        if inputs["sam_type"] != self.model.sam_type:
            self.model.sam.build_model(inputs["sam_type"])

        image_pil = Image.open(BytesIO(inputs["image_bytes"])).convert("RGB")
        results = self.model.predict(
            images_pil=[image_pil],
            texts_prompt=[inputs["text_prompt"]],
            box_threshold=inputs["box_threshold"],
            text_threshold=inputs["text_threshold"],
        )[0]

        if results["masks"] is None or len(results["masks"]) == 0:
            return image_pil, {
                "labels": [],
                "scores": [],
                "boxes": [],
                "masks": [],
            }

        encoded_masks = [rle_encode_binary_mask(mask) for mask in results["masks"]]
        result_dict = {
            "labels": results["labels"],
            "scores": results["scores"].tolist(),
            "boxes": [b.tolist() for b in results["boxes"]],
            "masks": encoded_masks,
        }

        image_np = np.asarray(image_pil)
        annotated = draw_image(
            image_np,
            results["masks"],
            results["boxes"],
            results["scores"],
            results["labels"],
        )
        annotated_img = Image.fromarray(np.uint8(annotated)).convert("RGB")

        return annotated_img, result_dict


class LangSAMImageAPI(BaseLangSAMAPI):
    def predict(self, inputs: dict) -> dict:
        annotated_img, _ = self.run_inference(inputs)
        return {"output_image": annotated_img}

    def encode_response(self, output: dict) -> Response:
        buffer = BytesIO()
        output["output_image"].save(buffer, format="PNG")
        buffer.seek(0)
        return Response(content=buffer.getvalue(), media_type="image/png")


class LangSAMJSONAPI(BaseLangSAMAPI):
    def predict(self, inputs: dict) -> dict:
        _, json_output = self.run_inference(inputs)
        return json_output

    def encode_response(self, output: dict) -> Response:
        return Response(content=json.dumps(output), media_type="application/json")

image_api = LangSAMImageAPI(api_path="/predict")
json_api = LangSAMJSONAPI(api_path="/predict_json")

server = ls.LitServer([image_api, json_api])

if __name__ == "__main__":
    server.run(port=PORT)
