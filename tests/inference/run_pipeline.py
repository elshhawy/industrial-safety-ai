import cv2
import time

# =========================
# Video Input
# =========================
# source = 0  # Camera
source = "videos/test_video_1.mp4"
cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# =========================
# Video Information
# =========================
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
source_fps = cap.get(cv2.CAP_PROP_FPS)

if source_fps <= 0:
    source_fps = 30.0

frame_time = 1.0 / source_fps

print("Video Input")
print("----------------")
print("Width:", width)
print("Height:", height)
print("Source FPS:", source_fps)


# =========================
# Processing FPS
# =========================
previous_time = time.perf_counter()

# =========================
# Main Pipeline
# =========================
while True:

    start_time = time.perf_counter()

    # 1. Capture
    ret, frame = cap.read()

    if not ret:
        break


    # 2. Processing
    # هنا سنضع YOLO لاحقًا
    processed_frame = frame


    # 3. Calculate Processing FPS
    current_time = time.perf_counter()
    processing_time = current_time - previous_time

    if processing_time > 0:
        processing_fps = 1.0 / processing_time
    else:
        processing_fps = 0

    previous_time = current_time


    # 4. Display Information

    cv2.putText(
        processed_frame,
        f"Source FPS: {source_fps:.1f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        processed_frame,
        f"Processing FPS: {processing_fps:.1f}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        processed_frame,
        f"Resolution: {width}x{height}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    # 5. Display
    cv2.imshow(
        "Real-Time Processing Pipeline",
        processed_frame
    )


    # 6. Maintain Original Video Speed

    elapsed = time.perf_counter() - start_time

    remaining_time = frame_time - elapsed

    if remaining_time > 0:
        time.sleep(remaining_time)


    # 7. Exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================
# Release
# =========================
cap.release()
cv2.destroyAllWindows()