# DanceRoom-2

# PREPROCESS

Run UniDepth first:  
https://github.com/atonalfreerider/UniDepth  
UniDepth model:  
https://huggingface.co/lpiccinelli/unidepth-v2-vitl14

Outputs:
- /depth folder of .npz
- wall floor line json

And then run DPVO:  
https://github.com/atonalfreerider/DPVO  
DPVO model:  
https://www.dropbox.com/s/nap0u8zslspdwm4/models.zip

Outputs:
- visual odometry json, where zoom and Z are conflated

Beat_This:  
https://github.com/atonalfreerider/beat_this

Beat_This model:  
https://cloud.cp.jku.at/index.php/s/7ik4RrBKTS273gp/download?path=%2F&files=final0.ckpt

Output:
- rhythm json

From main_process

# DANCER TRACKING

YOLO runs pose estimation and IoU tracking for video.

YOLO model:  
https://docs.ultralytics.com/tasks/pose/#models

Segmentation for later use in 3D display
SAM2_b model:  
https://docs.ultralytics.com/models/sam-2/#how-to-use-sam-2-versatility-in-image-and-video-segmentation

DeepFace for dancer identification

Face ID Model:  
https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5

# ROOM TRACKING

dance_room_tracker
- user camera position mid video, and room dimensions
- user warp input
- output warped, normalized poses, and warped visual odometry

Outputs are sent to:

GVHMR:  
https://github.com/atonalfreerider/GVHMR

GVHMR model:  
https://drive.google.com/drive/folders/10sEef1V_tULzddFxzCmDUpsIqfv7eP-P?usp=drive_link

stitched 3D model for lead and follow are ground ankle anchored

