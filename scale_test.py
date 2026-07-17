"""Stand-alone scale test: does YOLO detect the insect better when it appears
SMALLER in the frame?

Hypothesis: the VespAI model was trained with a larger camera distance, so a
hornet that fills a lot of our close-up frame is "too big" and mis-/under-
classified, while smaller apparent sizes (like a wasp, or a hornet seen from
farther) are recognised well.

The test does NOT change the pipeline. It takes one image, repeatedly shrinks
the whole frame by a factor and pads it back to the original size (= simulating
more camera distance: the insect gets smaller, more background around it), runs
YOLO on each version and prints the detections. If EH confidence rises as the
factor drops, the hypothesis holds.

Run ON THE PI (where torch works):
    python scale_test.py detections/events/<...>/frame.jpg
    python scale_test.py some_hornet.jpg --factors 1.0 0.7 0.5 0.35 0.25
Annotated versions are written next to a --out dir for visual inspection.
"""

import argparse
import os
import cv2
import numpy as np

from detection import load_model, run_detection


def shrink_and_pad(img, factor, fill):
    """Shrink the whole image by `factor` and pad back to original size with
    `fill` colour, centred -> the insect keeps its shape but occupies a smaller
    fraction of the frame (as if the camera were 1/factor farther away)."""
    h, w = img.shape[:2]
    sw, sh = max(1, int(w * factor)), max(1, int(h * factor))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)

    canvas = np.full((h, w, 3), fill, dtype=np.uint8)
    y0, x0 = (h - sh) // 2, (w - sw) // 2
    canvas[y0:y0 + sh, x0:x0 + sw] = small
    return canvas


def label_of(det):
    return "AH" if det.get("class_id") == 1 else "EH"


def main():
    ap = argparse.ArgumentParser(description="YOLO apparent-size sensitivity test")
    ap.add_argument("image", help="Path to a test image containing the insect")
    ap.add_argument(
        "--factors", type=float, nargs="+",
        default=[1.0, 0.8, 0.65, 0.5, 0.4, 0.3, 0.22],
        help="Shrink factors to try (1.0 = original size)",
    )
    ap.add_argument("--conf", type=float, default=0.05,
                    help="Show detections down to this confidence (default 0.05 to reveal weak hits)")
    ap.add_argument("--out", default="detections/scale_test", help="Dir for annotated outputs")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")

    os.makedirs(args.out, exist_ok=True)

    model = load_model()
    model.conf = args.conf  # reveal weak detections that the 0.93 event gate would hide

    # Fill colour = median of the frame (dominated by the green backdrop), so the
    # padding looks like more of the same background rather than black bars.
    fill = np.median(img.reshape(-1, 3), axis=0)

    frame_area = img.shape[0] * img.shape[1]
    print(f"\nImage: {args.image}  ({img.shape[1]}x{img.shape[0]})")
    print(f"{'factor':>7} | {'best det':>28} | all detections (label conf area%)")
    print("-" * 90)

    for f in args.factors:
        test = shrink_and_pad(img, f, fill) if f != 1.0 else img.copy()
        dets = run_detection(test, model)
        dets.sort(key=lambda d: d.get("confidence", 0.0), reverse=True)

        parts = []
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            area_pct = 100.0 * (x2 - x1) * (y2 - y1) / frame_area
            parts.append(f"{label_of(d)} {d['confidence']:.2f} {area_pct:.1f}%")

        best = parts[0] if parts else "(none)"
        print(f"{f:>7.2f} | {best:>28} | {'  '.join(parts) if parts else '-'}")

        # Save an annotated copy for visual inspection.
        vis = test.copy()
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            color = (0, 0, 255) if label_of(d) == "AH" else (0, 200, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"{label_of(d)} {d['confidence']:.2f}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imwrite(os.path.join(args.out, f"scale_{f:.2f}.jpg"), vis)

    print(f"\nAnnotated images written to: {args.out}")
    print("Read the table top-down: if EH appears / its confidence rises as the")
    print("factor drops, the insect was simply too big in the frame at 1.0.\n")


if __name__ == "__main__":
    main()
