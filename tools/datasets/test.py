from utils import extract_frames
frames = extract_frames("/shared/fangchen/junrui/Open-Sora/dataset/droid_480p24fps/AUTOLab/success/2023-07-07/Fri_Jul__7_09:42:23_2023/recordings/MP4/18026681.mp4",
               points=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
               )
for idx, frame in enumerate(frames):
    frame.save(f"dataloader/frame{idx}.jpg")
