from ultralytics import YOLO

from config import  MODELS


def getmodel(modelname):
  model = None
  if modelname in ["MODEL1", "MODEL2"]:
    model =  YOLO(MODELS[modelname])
  return model
   