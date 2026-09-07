import os
import shutil

def main():
    src_dir = "data/extracted_hf_dataset/cricketshot/train"
    dest_dir = "data/extracted_hf_dataset/cricketshot/tiny_train"

    if not os.path.exists(src_dir):
        print(f"Source directory {src_dir} does not exist.")
        return

    os.makedirs(dest_dir, exist_ok=True)

    classes = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]

    total_copied = 0
    for cls in classes:
        src_cls_dir = os.path.join(src_dir, cls)
        dest_cls_dir = os.path.join(dest_dir, cls)
        
        os.makedirs(dest_cls_dir, exist_ok=True)
        
        videos = [f for f in os.listdir(src_cls_dir) if os.path.isfile(os.path.join(src_cls_dir, f)) and not f.startswith('.')]
        videos = sorted(videos)[:2]
        
        for video in videos:
            src_video_path = os.path.join(src_cls_dir, video)
            dest_video_path = os.path.join(dest_cls_dir, video)
            shutil.copy2(src_video_path, dest_video_path)
            total_copied += 1
            
    print(f"Setup complete. Copied {total_copied} videos to {dest_dir}.")

if __name__ == "__main__":
    main()
