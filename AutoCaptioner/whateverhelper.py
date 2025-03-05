import os
import json
import torch
import h5py
import numpy as np
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample


def generate_training_data(dataset_name, json_annotations_path, images_directory, captions_per_image, 
                           min_word_occurrences, output_directory, max_caption_length=100):
    """
    Generates training data for an image captioning model.

    :param dataset_name: Name of the dataset (e.g., 'coco', 'flickr8k')
    :param json_annotations_path: Path to the JSON file with image captions
    :param images_directory: Directory containing images
    :param captions_per_image: Number of captions per image
    :param min_word_occurrences: Minimum frequency of a word to be included in the vocabulary
    :param output_directory: Directory where processed data will be saved
    :param max_caption_length: Maximum allowed length of a caption
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Load annotation file
    with open(json_annotations_path, 'r') as j:
        data = json.load(j)

    train_images, train_captions = [], []
    val_images, val_captions = [], []
    test_images, test_captions = [], []
    word_freq = Counter()

    for img in data['images']:
        captions = [c['tokens'] for c in img['sentences'] if len(c['tokens']) <= max_caption_length]
        if len(captions) == 0:
            continue

        path = os.path.join(images_directory, img['filepath'], img['filename']) \
            if dataset_name == 'coco' else os.path.join(images_directory, img['filename'])

        if img['split'] in {'train', 'restval'}:
            train_images.append(path)
            train_captions.append(captions)
        elif img['split'] == 'val':
            val_images.append(path)
            val_captions.append(captions)
        elif img['split'] == 'test':
            test_images.append(path)
            test_captions.append(captions)

        for caption in captions:
            word_freq.update(caption)

    # Create vocabulary
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_occurrences]
    word_map = {w: i+1 for i, w in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    base_filename = f"{dataset_name}_{captions_per_image}_cap_per_img_{min_word_occurrences}_min_word_freq"
    
    with open(os.path.join(output_directory, f"WORDMAP_{base_filename}.json"), 'w') as j:
        json.dump(word_map, j)

    seed(123)
    for image_set, caption_set, split in [(train_images, train_captions, 'TRAIN'),
                                          (val_images, val_captions, 'VAL'),
                                          (test_images, test_captions, 'TEST')]:

        with h5py.File(os.path.join(output_directory, f"{split}_IMAGES_{base_filename}.hdf5"), 'a') as h:
            h.attrs['captions_per_image'] = captions_per_image
            images = h.create_dataset('images', (len(image_set), 3, 256, 256), dtype='uint8')

            enc_captions, caplens = [], []

            for i, path in enumerate(tqdm(image_set, desc=f"Processing {split} data")):
                img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)  # Fake image placeholder

                if os.path.exists(path):
                    img = np.array(Image.open(path).resize((256, 256))).transpose(2, 0, 1)

                images[i] = img

                sampled_captions = caption_set[i] if len(caption_set[i]) >= captions_per_image else \
                    caption_set[i] + [choice(caption_set[i]) for _ in range(captions_per_image - len(caption_set[i]))]

                for c in sampled_captions:
                    enc_captions.append([word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] +
                                        [word_map['<end>']] + [word_map['<pad>']] * (max_caption_length - len(c)))
                    caplens.append(len(c) + 2)

            with open(os.path.join(output_directory, f"{split}_CAPTIONS_{base_filename}.json"), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_directory, f"{split}_CAPLENS_{base_filename}.json"), 'w') as j:
                json.dump(caplens, j)


def apply_gradient_clipping(optimizer, clip_value):
    """
    Clips gradients to prevent explosion of gradients.

    :param optimizer: Optimizer with gradients
    :param clip_value: Maximum gradient value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-clip_value, clip_value)


def store_model_checkpoint(dataset_name, epoch, encoder, decoder, encoder_optimizer, decoder_optimizer, bleu4, is_best):
    """
    Saves model checkpoint.

    :param dataset_name: Name of the dataset
    :param epoch: Current epoch number
    :param encoder: Encoder model
    :param decoder: Decoder model
    :param encoder_optimizer: Encoder optimizer
    :param decoder_optimizer: Decoder optimizer
    :param bleu4: BLEU-4 score for the current epoch
    :param is_best: Whether this is the best model so far
    """
    checkpoint_data = {
        'epoch': epoch,
        'bleu-4': bleu4,
        'encoder': encoder,
        'decoder': decoder,
        'encoder_optimizer': encoder_optimizer,
        'decoder_optimizer': decoder_optimizer
    }
    
    checkpoint_filename = f"checkpoint_{dataset_name}.pth.tar"
    torch.save(checkpoint_data, checkpoint_filename)
    
    if is_best:
        torch.save(checkpoint_data, f"BEST_{checkpoint_filename}")


class AverageMeter:
    """
    Keeps track of average, sum, and count of a metric.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: Optimizer whose learning rate needs to be adjusted
    :param shrink_factor: Factor to multiply the learning rate by
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] *= shrink_factor


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy.

    :param scores: Model scores
    :param targets: True labels
    :param k: Top-k value
    :return: Top-k accuracy
    """
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    return correct.view(-1).float().sum().item() * (100.0 / batch_size)
