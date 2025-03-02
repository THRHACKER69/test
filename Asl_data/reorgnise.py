import os
import shutil
import pandas as pd

# Paths
csv_files = ["train.csv", "val.csv", "test.csv"]  # List of CSV files
raw_videos_dir = "../Asl_data/Videos/raw_videos"  # Folder containing all original videos
organized_videos_dir = "../Asl_data/Videos/organized_videos"  # Output folder

# Create the organized_videos directory if not exists
os.makedirs(organized_videos_dir, exist_ok=True)

# Process each CSV file
for csv_file in csv_files:
    df = pd.read_csv(f"../Asl_data/CSV_Files/{csv_file}")

    for _, row in df.iterrows():
        video_filename = row["Video file"].strip()
        gloss_label = row["Gloss"].strip()

        # Create a folder for the gloss if it doesn't exist
        gloss_folder = os.path.join(organized_videos_dir, gloss_label)
        os.makedirs(gloss_folder, exist_ok=True)

        # Define source and destination paths
        src_path = os.path.join(raw_videos_dir, video_filename)
        dst_path = os.path.join(gloss_folder, video_filename)

        # Move the file if it exists
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
        else:
            print(f"Warning: {video_filename} not found in raw_videos.")

print("✅ Video organization completed!")
