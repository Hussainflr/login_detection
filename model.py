from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-cls.yaml")  # build a new model from YAML
model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11n-cls.yaml").load("yolo11n-cls.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="./data", epochs=100, imgsz=64)