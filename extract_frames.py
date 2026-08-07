import cv2
import os

sims = ['rossler', 'julia', 'epicycloid', 'random_walk', 'standing_wave']
artifact_dir = 'C:\\Users\\Aryan\\.gemini\\antigravity\\brain\\02ecd8e5-f0ac-4e84-a4bb-e91012038d0a'

for sim in sims:
    vid_path = os.path.join(artifact_dir, f'{sim}.mp4')
    if not os.path.exists(vid_path):
        print(f"Skipping {sim}, no video found.")
        continue
        
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print(f"Could not open {vid_path}")
        continue
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Extracting {sim} ({total_frames} frames)...")
    
    start_frame = int(total_frames * 0.1)
    mid_frame = int(total_frames * 0.5)
    end_frame = int(total_frames * 0.9)
    
    for label, f_idx in [('start', start_frame), ('mid', mid_frame), ('end', end_frame)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if ret:
            out_path = os.path.join(artifact_dir, f'{sim}_{label}.jpg')
            cv2.imwrite(out_path, frame)
            
    cap.release()
