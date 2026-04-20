from pathlib import Path
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(r"D:\Python\sam_project")
REPO = ROOT / "segment-anything"
sys.path.insert(0, str(REPO))

from segment_anything import sam_model_registry, SamPredictor

CHECKPOINT = ROOT / "models" / "sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"


def show_mask(mask, ax, color=(30/255, 144/255, 255/255, 0.45)):
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * np.array(color).reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=220):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    if len(pos_points):
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='lime', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    if len(neg_points):
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_sam_on_image.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print("Could not read image")
        sys.exit(1)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    sam = sam_model_registry[MODEL_TYPE](checkpoint=str(CHECKPOINT))
    sam.to(device="cpu")
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    input_point = np.array([[w // 2, h // 2]])
    input_label = np.array([1])

    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    best_idx = int(np.argmax(scores))
    best_mask = masks[best_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image)
    show_mask(best_mask, ax)
    show_points(input_point, input_label, ax)
    ax.set_title(f"SAM test | score={scores[best_idx]:.4f}")
    ax.axis('off')

    out_path = out_dir / f"{image_path.stem}_sam_result.png"
    fig.savefig(out_path, bbox_inches='tight', dpi=160)
    plt.close(fig)

    print(f"Saved result: {out_path}")
    print(f"Point used: ({w // 2}, {h // 2})")
    print("Note: script uses one positive point at image center.")


if __name__ == "__main__":
    main()

