import cv2

# 0 = Camera
# أو اكتب مسار الفيديو بدل 0
source = "videos/test_video_1.mp4"

cap = cv2.VideoCapture(source)

# التأكد أن المصدر اتفتح
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# معلومات المصدر
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Video Input Information")
print("-----------------------")
print("Source:", source)
print("Width:", width)
print("Height:", height)
print("FPS:", fps)
print("Total Frames:", total_frames)

if fps > 0:
    duration = total_frames / fps
    print("Duration:", round(duration, 2), "seconds")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Video Input", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()