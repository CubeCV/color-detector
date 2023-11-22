from ultralytics import YOLO
import argparse


def parse_option():
    parser = argparse.ArgumentParser("Validate color detector model", add_help=False)
    parser.add_argument("--model", type=str, required=True, metavar="FILE", help="Path to model")
    
    args = parser.parse_args()
    
    return args


def main(model_path):
    model = YOLO(model_path)

    # Evaluate the model on val set
    metrics = model.val()


if __name__ == "__main__":
    args = parse_option()
    main(args.model)
