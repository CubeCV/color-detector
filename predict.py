from ultralytics import YOLO
import argparse


def parse_option():
    parser = argparse.ArgumentParser("Predict with color detector model", add_help=False)
    parser.add_argument("--model", type=str, required=True, metavar="FILE", help="Path to model")
    parser.add_argument("--image", type=str, required=True, metavar="FILE", help="Path to image")
    
    args = parser.parse_args()

    return args


def main(model_path, image_path):
    model = YOLO(model_path)

    # Predict on image
    results = model(image_path, save=True)


if __name__ == "__main__":
    args = parse_option()
    main(args.model, args.image)
