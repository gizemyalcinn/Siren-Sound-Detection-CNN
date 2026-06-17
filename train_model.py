#Imports
import os
import pandas as pd
import librosa
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import soundfile as sf
import subprocess
import shutil
import json
import urllib.request

# AudioSet folder
os.makedirs("audioset", exist_ok=True)

balanced_url = "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv"
unbalanced_url = "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv"

balanced_path = "audioset/balanced_train_segments.csv"
unbalanced_path = "audioset/unbalanced_train_segments.csv"

# If files doesn't exist download
if not os.path.exists(balanced_path):
    urllib.request.urlretrieve(balanced_url, balanced_path)
    print("Balanced downloaded.")

if not os.path.exists(unbalanced_path):
    urllib.request.urlretrieve(unbalanced_url, unbalanced_path)
    print("Unbalanced downloaded.")

#Paths
URBAN_ROOT = "datasets/UrbanSound8K"
KAGGLE_ROOT = "datasets/sounds"

URBAN_METADATA_PATH = os.path.join(URBAN_ROOT, "metadata", "UrbanSound8K.csv")
URBAN_AUDIO_ROOT = os.path.join(URBAN_ROOT, "audio")

print("Urban metadata exists:", os.path.exists(URBAN_METADATA_PATH))
print("Urban audio exists:", os.path.exists(URBAN_AUDIO_ROOT))
print("Kaggle root exists:", os.path.exists(KAGGLE_ROOT))

print("Urban dataset:", os.path.exists(URBAN_ROOT))
print("Kaggle dataset:", os.path.exists(KAGGLE_ROOT))
print("Ambulance folder:", os.path.exists(os.path.join(KAGGLE_ROOT,"ambulance")))
print("Firetruck folder:", os.path.exists(os.path.join(KAGGLE_ROOT,"firetruck")))
print("Traffic folder:", os.path.exists(os.path.join(KAGGLE_ROOT,"traffic")))

#Feature Extraction
def extract_log_mel_fixed(file_path, duration=4, sr=16000, n_mels=128):

    audio, sr = librosa.load(file_path, sr=sr)

    target_length = duration * sr

    # If audio is short -> padding
    if len(audio) < target_length:
        pad_width = target_length - len(audio)
        audio = np.pad(audio, (0, pad_width))

    # If audio is long -> cropping
    else:
        audio = audio[:target_length]

    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels
    )

    log_mel = librosa.power_to_db(mel_spec)

    return log_mel

#UrbanSound8K Load
urban_df = pd.read_csv(URBAN_METADATA_PATH)

urban_df["label"] = (urban_df["class"] == "siren").astype(int)

def get_audio_path(row):
    fold = row["fold"]
    file_name = row["slice_file_name"]
    return os.path.join(URBAN_AUDIO_ROOT, f"fold{fold}", file_name)

urban_df["filepath"] = urban_df.apply(get_audio_path, axis=1)

train_df = urban_df[urban_df["fold"] <= 8].copy()
val_df_urban = urban_df[urban_df["fold"] == 9].copy()
test_df_urban = urban_df[urban_df["fold"] == 10].copy()

print("Urban train:", len(train_df))
print("Urban val:", len(val_df_urban))
print("Urban test:", len(test_df_urban))
print(train_df["label"].value_counts())

urban_train_df = train_df[["filepath", "label", "class"]].copy()
urban_train_df["source"] = "urban"

urban_siren_df = urban_train_df[urban_train_df["label"] == 1][["filepath", "label", "source"]].copy()
urban_non_siren_df = urban_train_df[urban_train_df["label"] == 0][["filepath", "label", "source"]].copy()

print("Urban siren:", len(urban_siren_df))
print("Urban non-siren:", len(urban_non_siren_df))

#Kaggle Load
ambulance_dir = os.path.join(KAGGLE_ROOT, "ambulance")
firetruck_dir = os.path.join(KAGGLE_ROOT, "firetruck")
traffic_dir = os.path.join(KAGGLE_ROOT, "traffic")

ambulance_files = [
    os.path.join(ambulance_dir, f)
    for f in os.listdir(ambulance_dir)
    if f.lower().endswith(".wav")
]

firetruck_files = [
    os.path.join(firetruck_dir, f)
    for f in os.listdir(firetruck_dir)
    if f.lower().endswith(".wav")
]

traffic_files = [
    os.path.join(traffic_dir, f)
    for f in os.listdir(traffic_dir)
    if f.lower().endswith(".wav")
]

kaggle_siren_df = pd.DataFrame({
    "filepath": ambulance_files + firetruck_files,
    "label": 1,
    "source": "kaggle"
})

kaggle_traffic_df = pd.DataFrame({
    "filepath": traffic_files,
    "label": 0,
    "source": "kaggle"
})

print("Kaggle siren:", len(kaggle_siren_df))
print("Kaggle traffic:", len(kaggle_traffic_df))

#Audioset Helpers
def read_audioset_csv(csv_path):
    df = pd.read_csv(
        csv_path,
        sep=", ",
        engine="python",
        skiprows=3,
        header=None,
        names=["YTID", "start_seconds", "end_seconds", "positive_labels"]
    )
    df["start_seconds"] = df["start_seconds"].astype(float)
    df["end_seconds"] = df["end_seconds"].astype(float)
    return df

def download_and_crop_segments(segment_df, output_folder, max_download=None):
    os.makedirs(output_folder, exist_ok=True)

    downloaded = 0
    failed = []

    rows = segment_df if max_download is None else segment_df.head(max_download)

    for _, row in rows.iterrows():
        video_id = str(row["YTID"]).strip()
        start = float(row["start_seconds"])
        end = float(row["end_seconds"])
        url = f"https://www.youtube.com/watch?v={video_id}"

        temp_template = f"{video_id}.%(ext)s"
        temp_wav = f"{video_id}.wav"
        out_wav = os.path.join(output_folder, f"{video_id}_{int(start)}_{int(end)}.wav")

        cmd_download = [
            "yt-dlp",
            "-x",
            "--audio-format", "wav",
            "--no-playlist",
            "-o", temp_template,
            url
        ]

        result_download = subprocess.run(
            cmd_download,
            capture_output=True,
            text=True
        )

        if result_download.returncode != 0 or not os.path.exists(temp_wav):
            failed.append((video_id, "download"))
            continue

        cmd_cut = [
            "ffmpeg",
            "-y",
            "-i", temp_wav,
            "-ss", str(start),
            "-to", str(end),
            out_wav
        ]

        result_cut = subprocess.run(
            cmd_cut,
            capture_output=True,
            text=True
        )

        if result_cut.returncode != 0:
            failed.append((video_id, "cut"))
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
            continue

        if os.path.exists(temp_wav):
            os.remove(temp_wav)

        downloaded += 1

    print(f"{output_folder} -> downloaded:", downloaded)
    print(f"{output_folder} -> failed:", len(failed))
    return downloaded, failed

def split_audioset_wavs_to_chunks(
    input_folder,
    output_folder,
    sr=16000,
    chunk_duration=4,
    step_duration=3
):
    os.makedirs(output_folder, exist_ok=True)

    wav_files = glob.glob(os.path.join(input_folder, "*.wav"))
    total_chunks = 0

    for wav_file in wav_files:
        audio, _ = librosa.load(wav_file, sr=sr)

        chunk_len = chunk_duration * sr
        step_len = step_duration * sr

        base_name = os.path.splitext(os.path.basename(wav_file))[0]
        chunk_count = 0

        for start in range(0, len(audio) - chunk_len + 1, step_len):
            end = start + chunk_len
            chunk = audio[start:end]

            out_path = os.path.join(output_folder, f"{base_name}_chunk{chunk_count}.wav")
            sf.write(out_path, chunk, sr)

            chunk_count += 1
            total_chunks += 1

    print("Input wav count:", len(wav_files))
    print("Output chunk count:", total_chunks)

def build_chunk_df(chunk_folder, source_name):
    chunk_files = glob.glob(os.path.join(chunk_folder, "*.wav"))
    return pd.DataFrame({
        "filepath": chunk_files,
        "label": 1,
        "source": source_name
    })

os.makedirs("audioset", exist_ok=True)

balanced_url = "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv"
unbalanced_url = "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv"

print("CSV dosyalari indirildi.")

#Audioset Balanced Siren

BALANCED_AUDIOSET_CSV = "audioset/balanced_train_segments.csv"
UNBALANCED_AUDIOSET_CSV = "audioset/unbalanced_train_segments.csv"

siren_label = "/m/04qvtq"

# Balanced
balanced_audioset_df = read_audioset_csv(BALANCED_AUDIOSET_CSV)
balanced_siren_df = balanced_audioset_df[
    balanced_audioset_df["positive_labels"].str.contains(siren_label, na=False)
].copy()

print("Balanced AudioSet siren segments:", len(balanced_siren_df))

download_and_crop_segments(
    balanced_siren_df,
    output_folder="audioset_balanced_wav",
    max_download=61
)

split_audioset_wavs_to_chunks(
    input_folder="audioset_balanced_wav",
    output_folder="audioset_balanced_chunks"
)

audioset_balanced_chunks_df = build_chunk_df(
    "audioset_balanced_chunks",
    "audioset_balanced"
)

print("Balanced chunk siren:", len(audioset_balanced_chunks_df))

if os.path.exists("audioset_balanced_wav"):
    shutil.rmtree("audioset_balanced_wav")

#Audioset Unbalanced Siren
unbalanced_audioset_df = read_audioset_csv(UNBALANCED_AUDIOSET_CSV)
unbalanced_siren_df = unbalanced_audioset_df[
    unbalanced_audioset_df["positive_labels"].str.contains(siren_label, na=False)
].copy()

print("Unbalanced AudioSet siren segments:", len(unbalanced_siren_df))

# İlk etapta 100 dene
sample_unbalanced = unbalanced_siren_df.sample(100, random_state=42)

download_and_crop_segments(
    sample_unbalanced,
    output_folder="audioset_unbalanced_wav",
    max_download=None
)

split_audioset_wavs_to_chunks(
    input_folder="audioset_unbalanced_wav",
    output_folder="audioset_unbalanced_chunks"
)

audioset_unbalanced_chunks_df = build_chunk_df(
    "audioset_unbalanced_chunks",
    "audioset_unbalanced"
)

print("Unbalanced chunk siren:", len(audioset_unbalanced_chunks_df))

if os.path.exists("audioset_unbalanced_wav"):
    shutil.rmtree("audioset_unbalanced_wav")

#Combine All
all_siren_df = pd.concat([
    urban_siren_df,
    kaggle_siren_df,
    audioset_balanced_chunks_df,
    audioset_unbalanced_chunks_df
], ignore_index=True)

all_non_siren_pool = pd.concat([
    urban_non_siren_df,
    kaggle_traffic_df
], ignore_index=True)

print("Total siren:", len(all_siren_df))
print(all_siren_df["source"].value_counts())

print("Non-siren pool:", len(all_non_siren_pool))
print(all_non_siren_pool["source"].value_counts())

#Balance Dataset
target = len(all_siren_df)

balanced_non_siren = all_non_siren_pool.sample(
    n=target,
    random_state=42
)

balanced_train_df = pd.concat([
    all_siren_df,
    balanced_non_siren
], ignore_index=True)

balanced_train_df = balanced_train_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print(balanced_train_df["label"].value_counts())
print(balanced_train_df["source"].value_counts())
print("Total balanced train:", len(balanced_train_df))

#Build X,Y
X = []
y = []

for _, row in balanced_train_df.iterrows():
    spec = extract_log_mel_fixed(row["filepath"])
    X.append(spec)
    y.append(row["label"])

X = np.array(X)
y = np.array(y)

print("X shape before channel:", X.shape)
print("y shape:", y.shape)

X = X[..., np.newaxis]

print("X shape after channel:", X.shape)

#Train-Validation Split
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train:", X_train.shape)
print("Validation:", X_val.shape)

#Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(128, 126, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

#Train
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop]
)

#Find best threshold
y_prob = model.predict(X_val).ravel()

thresholds = np.arange(0.1, 0.9, 0.01)
best_thr = 0
best_f1 = 0

for t in thresholds:
    y_pred_tmp = (y_prob > t).astype(int)
    f1 = f1_score(y_val, y_pred_tmp)

    if f1 > best_f1:
        best_f1 = f1
        best_thr = t

print("Best threshold:", best_thr)
print("Best F1:", best_f1)

threshold = float(round(best_thr, 2))

#Validation Report
y_pred = (y_prob > threshold).astype(int)

print("Threshold used:", threshold)
print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred))

#Save Model
os.makedirs("model", exist_ok=True)
model.save("model/siren_detector_model.keras")

#Save Threshold
with open("model/threshold.json", "w") as f:
    json.dump({"threshold": float(threshold)}, f)
print("Threshold kaydedildi:", threshold)



