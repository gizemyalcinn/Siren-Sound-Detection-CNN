# Siren Sound Detection Using CNN

This project presents a Convolutional Neural Network (CNN) based siren sound detection system developed as part of a graduation project focused on assistive technologies for individuals with disabilities.

## Project Purpose

The primary objective of this project is to detect emergency vehicle siren sounds and increase environmental awareness for hearing-impaired individuals.

## Dataset

The dataset was created by combining siren and non-siren audio samples obtained from:

- UrbanSound8K
- AudioSet
- Emergency Vehicle Siren Sounds

Audio recordings were preprocessed and transformed into log-Mel spectrogram representations before training.

## Data Preprocessing

The preprocessing pipeline includes:

- Audio resampling (16 kHz)
- Mono conversion
- Fixed-length segmentation
- Log-Mel spectrogram extraction
- Feature normalization

## Model Architecture

The CNN model consists of:

- Conv2D layers
- MaxPooling layers
- GlobalAveragePooling layer
- Dense layer
- Dropout layer
- Sigmoid output layer

## Training Configuration

- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Early Stopping: Enabled
- Classification Type: Binary Classification (Siren / Non-Siren)

## Results

The trained model achieved approximately **93% classification accuracy** during evaluation.

## Technologies

- Python
- TensorFlow / Keras
- Librosa
- NumPy
- Scikit-learn
- Flask

## Future Improvements

- Multi-class emergency sound detection
- Real-time edge deployment optimization
- Additional environmental sound categories

## Author

Gizem Yalçın
Computer Engineering
