import torch
import os
from sam2.build_sam import build_sam2_video_predictor
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image
import random

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

sam2_checkpoint = os.path.join(os.getcwd(), "experimental/sam2/checkpoints/sam2.1_hiera_large.pt")
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
video_dir = os.path.join(os.getcwd(), "experimental/tmp")

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)


cap = cv2.VideoCapture(os.path.join(os.getcwd(), "videos/alabama_clemson.mp4"))
ret, frame = cap.read()


model = YOLO('yolov8n.pt')
results = model(frame)

#points = torch.stack([box.reshape([2, 2]).sum(0).div(2) for box in results[0].boxes.xyxy])

inference_state = predictor.init_state(video_path=video_dir)
predictor.reset_state(inference_state)


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter("alabama_clemson_30s_clip_sam2_output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

for obj_id, box in enumerate(results[0].boxes.xyxy.unbind()):
    np_box = box.numpy()
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=obj_id,
        box=np_box,
    )
    
color_lookup = { idx:(random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)) for idx in range(results[0].boxes.xyxy.shape[0])}
    
# Each frame:
# 1. Use YOLO model to crop frames of people
# 2. 
    
# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results

for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    ret, frame = cap.read()
    if not ret:
        break
        
    
    
    
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }
    
    overlay = frame.copy()
    for id in out_obj_ids:
        mask_bool = video_segments[out_frame_idx][id].squeeze(0)
        mask_color = color_lookup[id]
        colored_mask = np.zeros_like(frame, dtype=np.uint8)
        colored_mask[:] = mask_color
        alpha = 0.5
        try:
            overlay[mask_bool] = cv2.addWeighted(frame[mask_bool], 1 - alpha, colored_mask[mask_bool], alpha, 0)
        except:
            continue
    
    cv2.imshow('frame', overlay)
    out.write(overlay)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
cap.release()
out.release()
cv2.destroyAllWindows()