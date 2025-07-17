import requests
import numpy as np
from PIL import Image
from io import BytesIO
from pycocotools import mask as mask_utils
import matplotlib.pyplot as plt
import supervision as sv

# ========== CONFIG ==========
IMAGE_PATH = "assets/perak.png"
TEXT_PROMPT = "lamppost. car. manhole. fire hydrant. electric pole."
API_URL = "http://localhost:57000/predict_json"
OUTPUT_PATH = "output_api_result.png"
# ============================

def draw_image(image_rgb, masks, xyxy, probs, labels):
    """Draw masks, boxes, and labels using supervision."""
    box_annotator = sv.BoxCornerAnnotator()
    label_annotator = sv.LabelAnnotator()
    mask_annotator = sv.MaskAnnotator()

    # Map labels to class ids
    unique_labels = list(set(labels))
    class_id_map = {label: idx for idx, label in enumerate(unique_labels)}
    class_ids = [class_id_map[label] for label in labels]

    detections = sv.Detections(
        xyxy=np.array(xyxy),
        mask=np.array(masks).astype(bool),
        confidence=np.array(probs),
        class_id=np.array(class_ids),
    )

    annotated_image = box_annotator.annotate(scene=image_rgb.copy(), detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    annotated_image = mask_annotator.annotate(scene=annotated_image, detections=detections)

    return annotated_image

def main():
    # Load image and send to API
    with open(IMAGE_PATH, "rb") as f:
        files = {"image": ("image.jpg", f, "image/jpeg")}
        data = {
            "sam_type": "sam2.1_hiera_small",
            "box_threshold": "0.3",
            "text_threshold": "0.25",
            "text_prompt": TEXT_PROMPT,
        }
        response = requests.post(API_URL, files=files, data=data)

    # Handle error
    if response.status_code != 200:
        print("Failed:", response.text)
        return

    results = response.json()
    if not results["masks"]:
        print("No masks returned")
        return

    # Decode inputs
    image_pil = Image.open(IMAGE_PATH).convert("RGB")
    image_np = np.array(image_pil)

    decoded_masks = [mask_utils.decode(rle) for rle in results["masks"]]
    boxes = results["boxes"]
    labels = results["labels"]
    scores = results["scores"]

    # Annotate
    annotated = draw_image(
        image_rgb=image_np,
        masks=decoded_masks,
        xyxy=boxes,
        probs=scores,
        labels=labels,
    )

    # Save and show
    Image.fromarray(annotated).save(OUTPUT_PATH)
    print(f"Saved annotated image to {OUTPUT_PATH}")

    # Print result info
    for i, label in enumerate(labels):
        print(f"\n[{i}] {label}")
        print(f" - Score: {scores[i]:.2f}")
        print(f" - Box: {boxes[i]}")
        print(f" - Mask area: {np.count_nonzero(decoded_masks[i])}")

    # Show with matplotlib
    plt.imshow(annotated)
    plt.axis("off")
    plt.title("LangSAM via API + supervision")
    plt.show()

if __name__ == "__main__":
    main()
