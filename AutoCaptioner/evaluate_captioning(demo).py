import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from image_caption_dataset import ImageCaptioningDataset
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
import json

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # Optimizes computation if input sizes are fixed

# Placeholder paths (Replace with actual paths)
dataset_directory = "your_dataset_path_here"
word_map_file = "your_word_map_path_here"
model_checkpoint = "your_model_checkpoint_here"


# Load model
checkpoint = torch.load(model_checkpoint, map_location=device)
decoder = checkpoint['decoder'].to(device).eval()
encoder = checkpoint['encoder'].to(device).eval()

# Load word map
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}


def compute_bleu_score(beam_size=3):
    """
    Evaluates the model using BLEU-4 score.

    :param beam_size: Beam width for generating captions
    :return: BLEU-4 score
    """
    # DataLoader
    loader = torch.utils.data.DataLoader(
        ImageCaptioningDataset(dataset_directory, "your_dataset_name", 'TEST', 
                               transform=transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                                   std=[0.229, 0.224, 0.225])])),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    references = []
    hypotheses = []

    for i, (image, caps, caplens, allcaps) in enumerate(tqdm(loader, desc=f"Evaluating (Beam={beam_size})")):
        image = image.to(device)
        encoder_out = encoder(image)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)
        encoder_out = encoder_out.view(1, -1, encoder_dim)

        k = beam_size
        prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)
        sequences = prev_words
        top_k_scores = torch.zeros(k, 1).to(device)
        complete_seqs = []
        complete_seqs_scores = []

        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        while True:
            embeddings = decoder.embedding(prev_words).squeeze(1)
            awe, _ = decoder.attention(encoder_out, h)
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

            prev_word_inds = top_k_words // len(word_map)
            next_word_inds = top_k_words % len(word_map)
            sequences = torch.cat([sequences[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)

            incomplete_inds = [ind for ind, word in enumerate(next_word_inds) if word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            if len(complete_inds) > 0:
                complete_seqs.extend(sequences[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)

            if k == 0 or step > 50:
                break

            sequences = sequences[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
            step += 1

        best_seq_index = complete_seqs_scores.index(max(complete_seqs_scores))
        best_seq = complete_seqs[best_seq_index]

        # References (Ground truth)
        img_caps = allcaps[0].tolist()
        img_captions = [[w for w in cap if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
                        for cap in img_caps]
        references.append(img_captions)

        # Hypothesis (Predicted caption)
        hypotheses.append([w for w in best_seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

        assert len(references) == len(hypotheses)

    # Compute BLEU-4 score
    bleu4_score = corpus_bleu(references, hypotheses)

    return bleu4_score


if __name__ == '__main__':
    beam_size = 5
    print(f"\nBLEU-4 score with beam size {beam_size}: {compute_bleu_score(beam_size):.4f}")
