Amharic Speech-to-Text (ASR) with Wav2Vec2




This repository provides an Automatic Speech Recognition (ASR) system for Amharic built using Facebook’s Wav2Vec2.0 model fine-tuned on Amharic speech datasets. The project aims to make Amharic speech recognition accessible and open-source, supporting low-resource language research and real-world applications.










Clone the repo and install dependencies:
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
Main dependencies:
Python 3.9+
PyTorch
Transformers
Datasets
Accelerate
jiwer (for WER)











📂 Dataset
This project uses Amharic speech datasets. You can replace or expand with your own data.
Example structure:
dataset/
 ├── train/
 │    ├── audio1.wav
 │    ├── audio2.wav
 │    └── ...
 ├── test/
 │    ├── audio1.wav
 │    └── ...
 └── metadata.csv   # transcripts, speaker info, etc.
