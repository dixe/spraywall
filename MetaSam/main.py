import torch
from segment_anything import sam_model_registry

import cv2
from segment_anything import SamAutomaticMaskGenerator



DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"

IMAGE_PATH = "sprayWall.jpg"

print(torch.cuda.is_available())
print(DEVICE)
exit()


print("loading")
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)

mask_generator = SamAutomaticMaskGenerator(sam)
image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


print("Generation masks")
result = mask_generator.generate(image_rgb)
