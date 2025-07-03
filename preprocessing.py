import os
import pandas as pd
import shutil
from pydub import AudioSegment


csv_path = "RecordingDetails.csv"
audio_folder = "data"  # folder containing the .m4a files
output_folder = "wavs"  # new folder for .wav files
os.makedirs(output_folder, exist_ok=True)

# --- LOAD CSV ---
df = pd.read_csv(os.path.join(audio_folder,csv_path))
df.columns = df.columns.str.strip()
df['StudentID'] = df['Audio File Name'].str.extract(r"^(\d+)")
df['StudentID'] = df['StudentID'].astype('int64')
unique_ids = sorted(df['StudentID'].unique())
student_id_map = {sid: i for i, sid in enumerate(unique_ids)}
df['StudentIndex'] = df['StudentID'].map(student_id_map)


story_map = {}
for student_id, group in df.groupby('StudentID'):
    unique_stories = sorted(group['Story Name'].unique())
    story_map[student_id] = {story: f"s{i+1}" for i, story in enumerate(unique_stories)}

def generate_clean_name(row):
    student_idx = row['StudentIndex']
    story_name = row['Story Name']
    para = 'p2' if 'Para2' in row['Audio File Name'] else 'p3'
    story_tag = story_map[row['StudentID']][story_name]
    return f"{student_idx}_{story_tag}_{para}.wav"

df['CleanedFileName'] = df.apply(generate_clean_name, axis=1)

for _, row in df.iterrows():
    original_path = os.path.join(audio_folder, row['Audio File Name'])
    new_path = os.path.join(output_folder, row['CleanedFileName'])

    if os.path.exists(original_path):
        try:
            audio = AudioSegment.from_file(original_path, format="m4a")
            audio.export(new_path, format="wav")
        except Exception as e:
            print(f"Error converting {original_path}: {e}")
    else:
        print(f"Missing file: {original_path}")

df.to_csv("RecordingDetails_Cleaned.csv", index=False)

print("Audio files renamed and saved to:", output_folder)
print("Updated CSV saved as: RecordingDetails_Cleaned.csv")
