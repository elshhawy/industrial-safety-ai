from ultralytics import YOLO


def main():
    model = YOLO("models/best.pt")

    metrics = model.val(
        data="training/data.yaml",
        split="val",
        conf=0.25,
        iou=0.6,
        plots=True,
        verbose=False,
    )

    print("\n" + "=" * 50)
    print("OVERALL METRICS")
    print("=" * 50)

    print(f"mAP50:      {metrics.box.map50:.3f}")
    print(f"mAP50-95:   {metrics.box.map:.3f}")
    print(f"Precision:  {metrics.box.mp:.3f}")
    print(f"Recall:     {metrics.box.mr:.3f}")

    print("\n" + "=" * 50)
    print("PER-CLASS METRICS")
    print("=" * 50)

    names = model.names

    for i, name in names.items():
        print(f"\n🔹 Class: {name}")
        print(f"   Precision: {metrics.box.p[i]:.3f}")
        print(f"   Recall:    {metrics.box.r[i]:.3f}")
        print(f"   mAP50:     {metrics.box.ap50[i]:.3f}")

    print("\n" + "=" * 50)
    print("Validation Complete")
    print("=" * 50)


if __name__ == "__main__":
    main()
