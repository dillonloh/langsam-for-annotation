import os
import cv2
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import supervision as sv
from pycocotools import mask as mask_utils

# ========== CONFIG ==========
INPUT_PATH = "assets/road.png"         # can be video.mp4 or image.jpg
IS_VIDEO = False                       # set to True for video
FRAME_INTERVAL = 15                    # used only if IS_VIDEO == True
TEXT_PROMPT = "traffic light. road sign. traffic sign"
API_URL = "http://localhost:57000/predict_json"
OUTPUT_DIR = "annotated_output"
# ============================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def draw_image(image_rgb, masks, xyxy, probs, labels):
    box_annotator = sv.BoxCornerAnnotator()
    label_annotator = sv.LabelAnnotator()
    mask_annotator = sv.MaskAnnotator()

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

def save_yolo_annotation(file_path, boxes, labels, image_size):
    unique_labels = list(set(labels))
    class_id_map = {label: idx for idx, label in enumerate(unique_labels)}
    width, height = image_size

    lines = []
    for label, box in zip(labels, boxes):
        class_id = class_id_map[label]
        x1, y1, x2, y2 = box

        x_center = ((x1 + x2) / 2) / width
        y_center = ((y1 + y2) / 2) / height
        w = (x2 - x1) / width
        h = (y2 - y1) / height

        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    with open(file_path, "w") as f:
        f.write("\n".join(lines))

def infer_and_annotate(image_pil, filename, image_size):
    # Convert to bytes
    buffered = BytesIO()
    image_pil.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()

    files = {"image": ("image.jpg", BytesIO(image_bytes), "image/jpeg")}
    data = {
        "sam_type": "sam2.1_hiera_small",
        "box_threshold": "0.3",
        "text_threshold": "0.25",
        "text_prompt": TEXT_PROMPT,
    }

    response = requests.post(API_URL, files=files, data=data)
    if response.status_code != 200:
        print(f"[ERROR] {filename}: {response.text}")
        return None

    results = response.json()
    if not results["masks"]:
        print(f"[INFO] {filename}: No detections.")
        return None

    decoded_masks = [mask_utils.decode(rle) for rle in results["masks"]]
    boxes = results["boxes"]
    labels = results["labels"]
    scores = results["scores"]

    image_np = np.array(image_pil)
    annotated = draw_image(image_np, decoded_masks, boxes, scores, labels)

    # Save image
    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    save_path = os.path.join(OUTPUT_DIR, f"{filename}_annotated.jpg")
    cv2.imwrite(save_path, annotated_bgr)

    # Save YOLO annotation
    txt_path = os.path.join(OUTPUT_DIR, f"{filename}.txt")
    save_yolo_annotation(txt_path, boxes, labels, image_size)

    print(f"[OK] Saved: {save_path}")
    print(f"[OK] Saved: {txt_path}")
    return annotated_bgr

def process_image(image_path):
    filename = os.path.splitext(os.path.basename(image_path))[0]
    image_pil = Image.open(image_path).convert("RGB")
    annotated_bgr = infer_and_annotate(image_pil, filename, image_pil.size)

    if annotated_bgr is not None:
        cv2.imshow("Annotated Image", annotated_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def process_video(video_path, frame_interval=15):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    saved_idx = 0

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    print(f"[INFO] Starting video stream from: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video stream.")
            break

        if frame_idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(frame_rgb).convert("RGB")
            filename = f"frame{saved_idx:05d}"
            annotated_bgr = infer_and_annotate(image_pil, filename, (frame.shape[1], frame.shape[0]))

            if annotated_bgr is not None:
                cv2.imshow("LangSAM Video Stream", annotated_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] Exiting early.")
                    break

            saved_idx += 1
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if IS_VIDEO:
        process_video(INPUT_PATH, FRAME_INTERVAL)
    else:
        process_image(INPUT_PATH)
