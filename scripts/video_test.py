from ultralytics import YOLO
import cv2

def main():
    # Path to input video
    video_path = "test.mp4"

    # Load trained model
    model = YOLO("models/best.pt")

    # Open video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video file")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

    # Initialize video writer for output
    out = cv2.VideoWriter(
        "output.mp4",
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    print("Running detection...")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        
        # Run inference on frame
        results = model(frame, conf=0.25, iou=0.5, imgsz=640)[0]

        violation_detected = False

        for box in results.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = model.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            color = (0, 255, 0)

            # Check for safety violations
            if label in ["NO-Hardhat", "NO-Safety Vest"]:
                color = (0, 0, 255)
                violation_detected = True

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label and confidence
            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        # Display alert text if violation detected
        if violation_detected:
            cv2.putText(
                frame,
                "SAFETY VIOLATION",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

        # Show frame
        cv2.imshow("Safety Detection", frame)

        # Save frame to output video
        out.write(frame)

        # Exit on ESC key
        if cv2.waitKey(1) == 27:
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Finished. Output saved as output.mp4")


if __name__ == "__main__":
    main()