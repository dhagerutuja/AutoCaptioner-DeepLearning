import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import argparse
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_caption_with_beam_search(encoder, decoder, img_path, word_dict, beam_width=5):
    """
    Generates an image caption using Beam Search.

    :param encoder: CNN encoder
    :param decoder: RNN/LSTM decoder
    :param img_path: Path to the input image
    :param word_dict: Vocabulary dictionary
    :param beam_width: Beam size for searching best sequence
    :return: Generated caption and attention weights
    """
    k = beam_width
    vocab_size = len(word_dict)

    # Read and preprocess image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = img.transpose(2, 0, 1) / 255.0  # Normalize
    img_tensor = torch.FloatTensor(img).to(device)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    img_tensor = transform(img_tensor).unsqueeze(0)  # Add batch dimension

    # Encode image
    encoder_out = encoder(img_tensor)
    enc_img_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)
    encoder_out = encoder_out.view(1, -1, encoder_dim).expand(k, -1, encoder_dim)

    # Start the beam search
    prev_words = torch.LongTensor([[word_dict['<start>']]] * k).to(device)
    sequences = prev_words
    top_k_scores = torch.zeros(k, 1).to(device)
    seq_alphas = torch.ones(k, 1, enc_img_size, enc_img_size).to(device)

    complete_seqs = []
    complete_seqs_scores = []

    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    while True:
        embeddings = decoder.embedding(prev_words).squeeze(1)
        awe, alpha = decoder.attention(encoder_out, h)
        alpha = alpha.view(-1, enc_img_size, enc_img_size)
        gate = decoder.sigmoid(decoder.f_beta(h))
        awe = gate * awe
        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))

        scores = decoder.fc(h)
        scores = F.log_softmax(scores, dim=1)
        scores = top_k_scores.expand_as(scores) + scores

        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

        prev_word_inds = top_k_words // vocab_size
        next_word_inds = top_k_words % vocab_size

        sequences = torch.cat([sequences[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
        seq_alphas = torch.cat([seq_alphas[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)], dim=1)

        incomplete_inds = [idx for idx, word in enumerate(next_word_inds) if word != word_dict['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        if len(complete_inds) > 0:
            complete_seqs.extend(sequences[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)

        if k == 0 or step > 50:
            break

        sequences = sequences[incomplete_inds]
        seq_alphas = seq_alphas[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        step += 1

    best_seq_idx = complete_seqs_scores.index(max(complete_seqs_scores))
    final_seq = complete_seqs[best_seq_idx]
    final_alphas = seq_alphas[best_seq_idx]

    return final_seq, final_alphas


def display_attention_overlay(image_path, caption_seq, attention_weights, id_to_word, smooth=True):
    """
    Visualizes attention overlay for the generated caption.

    :param image_path: Path to input image
    :param caption_seq: Generated caption sequence
    :param attention_weights: Attention weights for visualization
    :param id_to_word: Reverse word mapping (index to word)
    :param smooth: Apply smoothing to attention overlay
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [id_to_word[idx] for idx in caption_seq]

    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)
        plt.text(0, 1, words[t], color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)

        alpha_img = attention_weights[t, :].numpy()
        if smooth:
            alpha_img = cv2.resize(alpha_img, (14 * 24, 14 * 24))
        plt.imshow(alpha_img, alpha=0.8, cmap=cm.Greys_r)
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Image Caption')

    parser.add_argument('--image', '-i', help='Path to the image')
    parser.add_argument('--model', '-m', help='Path to the model')
    parser.add_argument('--word_map', '-wm', help='Path to word map JSON')
    parser.add_argument('--beam_size', '-b', default=5, type=int, help='Beam size for search')
    parser.add_argument('--no_smooth', dest='smooth', action='store_false', help='Disable smoothing for attention')

    args = parser.parse_args()

    # Load model
    checkpoint = torch.load(args.model, map_location=str(device))
    decoder = checkpoint['decoder'].to(device).eval()
    encoder = checkpoint['encoder'].to(device).eval()

    # Load word map
    with open(args.word_map, 'r') as file:
        word_map = json.load(file)
    rev_word_map = {v: k for k, v in word_map.items()}

    # Generate caption
    sequence, alphas = generate_caption_with_beam_search(encoder, decoder, args.image, word_map, args.beam_size)
    alphas = torch.FloatTensor(alphas)

    # Display results
    display_attention_overlay(args.image, sequence, alphas, rev_word_map, args.smooth)
