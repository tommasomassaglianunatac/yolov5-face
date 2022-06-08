import torch

model = torch.load('weights/yolov5s-face.pt', map_location='cpu')['model']
torch.save(model.state_dict(),'weights/yolov5s_state_dict.pt')