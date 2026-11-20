import cv2
import easyocr
from ultralytics import YOLO
from pathlib import Path

# Build all file paths relative to this Python file so the script works from
# any terminal location.
base_dir = Path(__file__).resolve().parent

# 1. Load models
# YOLO's COCO model finds common vehicles, while the optional custom model
# is intended to find license plates directly.
car_model_path = base_dir / "yolov8n.pt"
plate_model_path = base_dir / "license_plate_detector.pt"

# The COCO model is required because it locates the cars in the image.
if not car_model_path.is_file():
    raise FileNotFoundError(f"Vehicle model not found: {car_model_path}")

car_model = YOLO(str(car_model_path))  # Standard COCO model for vehicles

# Use the custom plate model when available; otherwise, use an OpenCV fallback.
plate_model = None
if plate_model_path.is_file():
    plate_model = YOLO(str(plate_model_path))  # Custom YOLO model for plates
else:
    print("Custom plate model not found; using OpenCV plate detection.")

# EasyOCR reads printed text; "en" supports English and "ar" supports Arabic.
reader = easyocr.Reader(["en", "ar"])  # Add 'ar' if reading Arabic/Tunisian plates

# Load image (or loop through video frames)
# Check the image path and ensure OpenCV loaded it before processing it.
image_path = base_dir / "car_photo.jpg"
if not image_path.is_file():
    raise FileNotFoundError(f"Image not found: {image_path}")

image = cv2.imread(str(image_path))
if image is None:
    raise ValueError(f"OpenCV could not read image: {image_path}")

# 2. Detect vehicles (COCO class 2 = car, 3 = motorcycle, 5 = bus, 7 = truck)
# Run vehicle detection first. The confidence value removes uncertain detections.
vehicle_results = car_model(
    image,
    classes=[2, 3, 5, 7],
    conf=0.5,
    verbose=False,
    device="cpu",
)

for box in vehicle_results[0].boxes:
    # Each YOLO box contains two opposite corners: top-left (x1, y1)
    # and bottom-right (x2, y2).
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # Clamp coordinates to the image edges to prevent invalid crop ranges.
    height, width = image.shape[:2]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        continue

    # Draw green box around the car
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Crop the car region to search for the plate inside it
    car_crop = image[y1:y2, x1:x2]

    # 3. Detect license plate within the cropped car
    # The plate boxes are stored as (left, top, right, bottom) coordinates.
    plate_boxes = []
    if plate_model is not None:
        # Preferred method: let the custom YOLO model find each plate.
        plate_results = plate_model(
            car_crop,
            conf=0.4,
            verbose=False,
            device="cpu",
        )
        for p_box in plate_results[0].boxes:
            px1, py1, px2, py2 = map(int, p_box.xyxy[0])
            plate_boxes.append((px1, py1, px2, py2))
    else:
        # Fallback method: convert to grayscale, reduce noise, and find edges.
        gray_crop = cv2.cvtColor(car_crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray_crop, 11, 17, 17)
        edges = cv2.Canny(blurred, 30, 200)

        # Contours outline the visible shapes in the edge image.
        contours, _ = cv2.findContours(
            edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates = []
        for contour in contours:
            # Keep shapes that look similar to a horizontal license plate.
            px, py, pw, ph = cv2.boundingRect(contour)
            aspect_ratio = pw / float(ph) if ph else 0
            if 2.0 <= aspect_ratio <= 7.0 and pw * ph >= car_crop.shape[0] * car_crop.shape[1] * 0.01:
                candidates.append((px, py, px + pw, py + ph))

        # Choose the three largest plausible candidates for OCR.
        plate_boxes = sorted(
            candidates,
            key=lambda box: (box[2] - box[0]) * (box[3] - box[1]),
            reverse=True,
        )[:3]

    for px1, py1, px2, py2 in plate_boxes:
        # Clamp plate coordinates to the cropped vehicle image.
        px1, px2 = max(0, px1), min(car_crop.shape[1], px2)
        py1, py2 = max(0, py1), min(car_crop.shape[0], py2)
        if px2 <= px1 or py2 <= py1:
            continue

        # Crop the license plate
        plate_crop = car_crop[py1:py2, px1:px2]

        # Convert to grayscale for better OCR accuracy
        gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

        # 4. Read text from the license plate
        # EasyOCR returns each detected text part and its confidence score.
        # This joins every text part into one plate string.
        ocr_result = reader.readtext(gray_plate)
        plate_text = " ".join([res[1] for res in ocr_result]).strip()

        if plate_text:
            print(f"Detected Plate: {plate_text}")
            # Draw blue box around plate and put text above the car
            cv2.rectangle(
                image,
                # Convert crop-relative coordinates back to full-image coordinates.
                (x1 + px1, y1 + py1),
                (x1 + px2, y1 + py2),
                (255, 0, 0),
                2,
            )
            cv2.putText(
                image,
                plate_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

# Always save the result so it is available even when the GUI is unsupported.
output_path = base_dir / "anpr_result.jpg"
if cv2.imwrite(str(output_path), image):
    print(f"Operation completed: saved result image to {output_path}")
else:
    raise IOError(f"Could not save result image: {output_path}")
