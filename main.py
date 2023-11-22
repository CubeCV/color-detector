from ultralytics import YOLO


def main():
    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(
                data='data.yaml',
                epochs=100,
                optimizer='Adam',
                fraction=0.8,
            )
    # Export the model to .pt file
    success = model.export()


if __name__ == "__main__":
    main()

