import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from captioning_model import ImageEncoder, CaptionDecoder
from image_caption_dataset import ImageCaptioningDataset
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import os
import json

# Placeholder paths (Replace with actual paths)
dataset_path = "your_dataset_path_here"
word_map_path = "your_word_map_path_here"
checkpoint_path = "your_checkpoint_path_here"

# Training Parameters
start_epoch = 0
num_epochs = 100  
batch_size = 32
workers = 1  
encoder_lr = 1e-4  
decoder_lr = 4e-4  
grad_clip = 5.  
alpha_c = 1.  
best_bleu4 = 0.  
print_freq = 100  
fine_tune_encoder = False  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


def train_model():
    """
    Train the Image Captioning Model.
    """
    global best_bleu4, start_epoch, fine_tune_encoder

    # Load word map
    with open(word_map_path, 'r') as j:
        word_map = json.load(j)

    # Load or initialize model
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch'] + 1
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()), 
                                                 lr=encoder_lr)
    else:
        decoder = CaptionDecoder(attention_dim=512, embed_dim=512, decoder_dim=512, vocab_size=len(word_map), dropout=0.5)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=decoder_lr)

        encoder = ImageEncoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()), 
                                             lr=encoder_lr) if fine_tune_encoder else None

    # Move to device
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # DataLoader
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        ImageCaptioningDataset(dataset_path, "your_dataset_name", 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        ImageCaptioningDataset(dataset_path, "your_dataset_name", 'VAL', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(decoder_optimizer, step_size=10, gamma=0.8)

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        if fine_tune_encoder:
            scheduler.step()

        train_one_epoch(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch)

        # Validation
        recent_bleu4 = validate(val_loader, encoder, decoder, criterion)

        # Save checkpoint
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)

        save_checkpoint("your_dataset_name", epoch, encoder, decoder, encoder_optimizer, decoder_optimizer, 
                        recent_bleu4, is_best)


def train_one_epoch(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Train model for one epoch.
    """
    decoder.train()
    encoder.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        targets = caps_sorted[:, 1:]
        scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        loss = criterion(scores, targets)
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        decoder_optimizer.zero_grad()
        if encoder_optimizer:
            encoder_optimizer.zero_grad()
        loss.backward()

        if grad_clip:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer:
                clip_gradient(encoder_optimizer, grad_clip)

        decoder_optimizer.step()
        if encoder_optimizer:
            encoder_optimizer.step()

        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        if i % print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})')


def validate(val_loader, encoder, decoder, criterion):
    """
    Validate the model.
    """
    decoder.eval()
    if encoder:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()
    references = []
    hypotheses = []

    with torch.no_grad():
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            if encoder:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            targets = caps_sorted[:, 1:]
            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            loss = criterion(scores, targets)
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)
            start = time.time()

        bleu4 = corpus_bleu(references, hypotheses)
        print(f'Validation: Loss {losses.avg:.4f}, Top-5 Accuracy {top5accs.avg:.3f}, BLEU-4 {bleu4:.4f}')
    
    return bleu4


if __name__ == '__main__':
    train_model()
