from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='data.yaml', epochs=100)

# Evaluate the model on val set
metrics = model.val()

# Export the model to .pt file
success = model.export()
