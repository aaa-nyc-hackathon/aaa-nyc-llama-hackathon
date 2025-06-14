import cv2
import numpy as np
import os


from sam2.build_sam import build_sam2
from sam2.build_sam import build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import torch




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

# Video reader
cap = cv2.VideoCapture(os.path.join(os.getcwd(), "videos/louisville_60s_clip.mp4"))
ret, prev_frame = cap.read()
gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

sam2_checkpoint = os.path.join(os.getcwd(), "experimental/sam2/checkpoints/sam2.1_hiera_large.pt")
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

width = 640
height = 360
max_area = width * height

mask_generator = SAM2AutomaticMaskGenerator(sam2,
                                            stability_score_thresh=0.95,
                                            stability_score_offset=0.1,
                                            points_per_side=8
                                           )
                
# NOTE: We use a single call of SAM full mask segmentation to locate the court
# We only extract the initial position of the court from the first frame of the video
# This will become problematic in production when providing videos that may not start with an ideal frame of the court                           
masks = mask_generator.generate(prev_frame)
masks.sort(key=lambda x: x['area'])
points = masks[-2]['point_coords']

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
video_dir = os.path.join(os.getcwd(), "experimental/sam2/tmp")
inference_state = predictor.init_state(video_path=video_dir)
predictor.reset_state(inference_state)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(os.path.join(os.getcwd(), "experimental/floor_detection/outputs/louisville_60s_clip_segmentation.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=0,
    obj_id=1,
    points=np.array(points, dtype=np.float32),
    labels=np.array([1 for _ in range(len(points))])
)

show_overlay = False # If writing to output, keep this false
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    ret, frame = cap.read()
    if not ret:
        break
        
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }
    
    overlay = np.zeros_like(frame, dtype=np.uint8)
    for id in out_obj_ids:
        mask_bool = video_segments[out_frame_idx][id].squeeze(0)
        mask_color = (0, 255, 0)
        colored_mask = np.zeros_like(frame, dtype=np.uint8)
        colored_mask[:] = mask_color
        alpha = 0.5
        try:
            if show_overlay:
                overlay[mask_bool] = cv2.addWeighted(frame[mask_bool], 1 - alpha, colored_mask[mask_bool], alpha, 0)
            else:
                overlay[mask_bool] = colored_mask[mask_bool]
        except:
            continue
    
    cv2.imshow('frame', overlay)
    out.write(overlay)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
        
cap.release()
out.release()