# AutoCaptioner-DeepLearning
About Me:
I am Rutuja Dhage, an AI & Data Science student interested in advanced machine learning and AI applications.

About the Project:
This project implements an AI-powered image captioning model using CNNs, LSTMs, and Attention Mechanism to generate accurate image descriptions.

Image Captioning with Attention Mechanism

Overview
This project implements an image captioning model using deep learning, specifically Show, Attend, and Tell architecture. It generates descriptive captions for images by leveraging a CNN-based Encoder (ResNet) and an LSTM-based Decoder with an Attention Mechanism.

Features
✔ Encoder-Decoder Architecture using CNN & LSTM
✔ Attention Mechanism to focus on relevant image regions
✔ Beam Search for better caption generation
✔ Pretrained ResNet-101 Encoder for feature extraction
✔ MSCOCO Dataset Support

Installation
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt

Usage
Training the Model
python train.py

Generating Captions
python caption.py --img="path/to/image.jpg" --model="path/to/checkpoint.pth.tar" --word_map="path/to/word_map.json"

Dataset
MSCOCO 2014 dataset for training
Preprocessing includes image resizing, tokenization, and vocabulary mapping

Results
✅ Generates captions with attention visualization
✅ Can handle real-world images effectively
