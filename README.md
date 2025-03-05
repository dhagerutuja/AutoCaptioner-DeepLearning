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

Installation
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt

Usage
Preparing dataset
python prepare_dataset.py

Training the Model
python train_model5.py


Generating Captions for an Image
python image_caption_generator_final.py --image "your_image_path_here" --model "your_checkpoint_path_here" --word_map "your_word_map_path_here"

Evaluating the Model
python evaluate_captioning.py

AutoCaptioner-DeepLearning/
│── for_Images/                     # Folder for storing images
│── Captioning_modelfinal.py         # Encoder-Decoder Model (CNN + LSTM + Attention)
│── evaluate_captioning(demo).py     # Evaluates model performance using BLEU
│── image_caption_dataset.py         # Custom PyTorch Dataset for training/testing
│── image_caption_generator_final.py # Generates captions for images
│── prepare_dataset.py               # Prepares dataset files for training/testing
│── train_model5.py                   # Trains the model using CNN + LSTM
│── whateverhelper.py                 # Utility functions (data processing, BLEU, etc.)
│── README.md                         # Project documentation



Results
✅ Generates captions with attention visualization
✅ Can handle real-world images effectively
