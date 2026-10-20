import cv2
import time
from ultralytics import YOLO


# =========================
# 1. Load Model
# =========================

model = YOLO("best.pt")

device = 0  # GPU


# =========================
# 2. Video Input
# =========================

#source = "videos/test_video_1.mp4"
source = 0  # Camera

cap = cv2.VideoCapture(source)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()


# =========================
# 3. Video Information
# =========================

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
source_fps = cap.get(cv2.CAP_PROP_FPS)

if source_fps <= 0:
    source_fps = 30.0

frame_time = 1.0 / source_fps


print("\nVideo Information")
print("-------------------------")
print("Resolution:", width, "x", height)
print("Source FPS:", source_fps)
print("GPU Device: CUDA:0")


# =========================
# 4. Performance Variables
# =========================

previous_frame_time = time.perf_counter()

frame_count = 0

total_inference_time = 0
total_processing_time = 0


# =========================
# 5. Main Pipeline
# =========================

while True:

    frame_start = time.perf_counter()


    # -------------------------
    # Read Frame
    # -------------------------

    ret, frame = cap.read()

    if not ret:
        break


    # -------------------------
    # YOLO Inference
    # -------------------------

    inference_start = time.perf_counter()

    results = model.predict(
        frame,
        device=device,
        verbose=False
    )

    inference_end = time.perf_counter()


    # Inference time
    inference_time = (
        inference_end - inference_start
    ) * 1000

    total_inference_time += inference_time


    # -------------------------
    # Draw Results
    # -------------------------

    processed_frame = results[0].plot()


    # -------------------------
    # Total Processing Time
    # -------------------------

    frame_end = time.perf_counter()

    total_time = frame_end - frame_start

    total_processing_time += total_time

    frame_count += 1


    # -------------------------
    # Calculate FPS
    # -------------------------

    if total_time > 0:
        processing_fps = 1 / total_time
    else:
        processing_fps = 0


    if inference_time > 0:
        inference_fps = 1000 / inference_time
    else:
        inference_fps = 0


    # -------------------------
    # Display Information
    # -------------------------

    cv2.putText(
        processed_frame,
        f"Source FPS: {source_fps:.1f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    cv2.putText(
        processed_frame,
        f"Inference: {inference_time:.1f} ms",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        processed_frame,
        f"Inference FPS: {inference_fps:.1f}",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        processed_frame,
        f"Processing FPS: {processing_fps:.1f}",
        (20, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


    # -------------------------
    # Display
    # -------------------------

    cv2.imshow(
        "YOLO Performance Test",
        processed_frame
    )


    # -------------------------
    # Maintain Video Speed
    # -------------------------

    elapsed = time.perf_counter() - frame_start

    remaining_time = frame_time - elapsed

    if remaining_time > 0:
        time.sleep(remaining_time)


    # -------------------------
    # Exit
    # -------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================
# 6. Final Performance
# =========================

cap.release()
cv2.destroyAllWindows()


# =========================
# 7. Print Final Results
# =========================

if frame_count > 0:

    average_inference = (
        total_inference_time / frame_count
    )

    average_processing = (
        total_processing_time / frame_count
    )

    average_processing_fps = (
        1 / average_processing
    )

    print("\nPerformance Results")
    print("=========================")
    print("Frames Processed:", frame_count)
    print(
        "Average Inference:",
        round(average_inference, 2),
        "ms"
    )
    print(
        "Average Inference FPS:",
        round(1000 / average_inference, 2)
    )
    print(
        "Average Processing FPS:",
        round(average_processing_fps, 2)
    )