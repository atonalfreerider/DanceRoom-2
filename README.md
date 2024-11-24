# DanceRoom-2

## TRACK PREPROCESS

Run DPVO:  
https://github.com/atonalfreerider/DPVO  
DPVO model:  
https://www.dropbox.com/s/nap0u8zslspdwm4/models.zip

Output:
- visual odometry json, where zoom and Z are conflated

Run Beat_This:  
https://github.com/atonalfreerider/beat_this

Beat_This model:  
https://cloud.cp.jku.at/index.php/s/7ik4RrBKTS273gp/download?path=%2F&files=final0.ckpt

Output:
- rhythm json

NOT CURRENTLY USING:  
Run UniDepth:  
https://github.com/atonalfreerider/UniDepth  
UniDepth model:  
https://huggingface.co/lpiccinelli/unidepth-v2-vitl14

Outputs:
- /depth folder of .npz
- wall floor line json


## DANCER TRACKING with MAIN PROCESS

YOLO runs pose estimation and IoU tracking for video.

YOLO model:  
https://docs.ultralytics.com/tasks/pose/#models


DeepFace for dancer identification

Face ID Model:  
https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5

User input for corrections  

## ROOM TRACKING

dance_room_tracker
- user camera position mid-video, and room dimensions
- user warp input
- output warped, normalized poses, and warped visual odometry

TODO: CoTracker for backline tracking and camera rotation and focus warp
Model:  
https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth

Ankles get grounded and output all_floor_ankles.json

NOT CURRENTLY USING: 
Segmentation for later use in 3D display
SAM2_b model:  
https://docs.ultralytics.com/models/sam-2/#how-to-use-sam-2-versatility-in-image-and-video-segmentation

## TRACK POST PROCESS

GVHMR:  
https://github.com/atonalfreerider/GVHMR

GVHMR model:  
https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD?usp=drive_link

output keypoints 3D json for lead and follow  


## RHYTHM PHYSICS

TODO


