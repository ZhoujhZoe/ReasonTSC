from momentfm import MOMENTPipeline
from torch.utils.data import DataLoader
from momentfm.data.classification_dataset import ClassificationDataset
import torch
from torch import nn
from pprint import pprint
import torch
from tqdm import tqdm
import numpy as np
import random
import os
import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MOMENT Classification Training')
    parser.add_argument('--num-class', type=int, default=5, help='Number of classes for classification')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--output-file', type=str, default='./MOMENT_Acoustic_window200.txt', 
                       help='Path to save evaluation results')
    parser.add_argument('--model-path', type=str, default='D:/Models/MOMENT/MOMENT-1-large',
                       help='Path to pretrained MOMENT model')
    return parser.parse_args()


args = parse_args()

model = MOMENTPipeline.from_pretrained(
    args.model_path,
    model_kwargs={
        'task_name': 'classification',
        'n_channels': 12,  # Number of input channels
        'num_class': args.num_class,  # Now passed as command line argument
        'freeze_encoder': False,  # Freeze the patch embedding layer
        'freeze_embedder': False,  # Freeze the transformer encoder
        'freeze_head': False,  # The linear forecasting head must be trained
        # NOTE: Disable gradient checkpointing to suppress the warning when linear probing the model as MOMENT encoder is frozen
        'enable_gradient_checkpointing': False,
        # Choose how embedding is obtained from the model: One of ['mean', 'concat']
        # Multi-channel embeddings are obtained by either averaging or concatenating patch embeddings 
        # along the channel dimension. 'concat' results in embeddings of size (n_channels * d_model), 
        # while 'mean' results in embeddings of size (d_model)
        'reduction': 'mean',
    },
    local_files_only=True,  # Whether or not to only look at local files (i.e., do not try to download the model).
)

model.init()


total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")  
print(f"Total parameters: {total_params/1e9:.2f}B") 
print(f"Total parameters: {total_params / 1e6:.2f}M")


def control_randomness(seed: int = 42):
    """Function to control randomness in the code."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
control_randomness(42)


def train_epoch(model, device, train_dataloader, criterion, optimizer, scheduler, reduction='mean'):
    """Train the model for one epoch."""
    model.to(device)
    model.train()
    losses = []

    for batch_idx, (batch_x, batch_masks, batch_labels) in enumerate(train_dataloader):
        optimizer.zero_grad()
        batch_x = batch_x.to(device).float()
        batch_labels = batch_labels.to(device).long()

        # Note that since MOMENT encoder is based on T5, it might experience numerical instability with float16
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32):
            output = model(x_enc=batch_x, reduction=reduction)
            loss = criterion(output.logits, batch_labels)
        loss.backward()

        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    
    avg_loss = np.mean(losses)
    return avg_loss


def evaluate_epoch(dataloader, model, criterion, device, phase='val', reduction='mean', output_file=args.output_file):
    """Evaluate the model on the given dataset."""
    model.eval()
    model.to(device)
    total_loss, total_correct = 0, 0
    idx = 0

    with open(output_file, 'w') as f:
        f.write("Batch_idx, Predicted_Label, True_Label, Logits\n") 

        with torch.no_grad():
            for batch_idx, (batch_x, batch_masks, batch_labels) in enumerate(dataloader):
                batch_x = batch_x.to(device).float()
                batch_labels = batch_labels.to(device).long()

                output = model(x_enc=batch_x, reduction=reduction)
                loss = criterion(output.logits, batch_labels)
                total_loss += loss.item()
                total_correct += (output.logits.argmax(dim=1) == batch_labels).sum().item()

                # Get predicted labels and logits
                predicted_labels = output.logits.argmax(dim=1).cpu().numpy()
                true_labels = batch_labels.cpu().numpy()
                logits = output.logits.cpu().numpy()

                for i in range(len(predicted_labels)):
                    f.write(f"{idx}, {predicted_labels[i]}, {true_labels[i]}, {logits[i].tolist()}\n")
                    idx = idx + 1
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / len(dataloader.dataset)
    return avg_loss, accuracy


train_dataset = ClassificationDataset(data_split='train')
test_dataset = ClassificationDataset(data_split='test')
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False, drop_last=False)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, total_steps=args.epochs * len(train_dataloader))
device = 'cuda'
reduction='mean'

for i in tqdm(range(args.epochs)):
    train_loss = train_epoch(model, device, train_dataloader, criterion, optimizer, scheduler)
    val_loss, val_accuracy = evaluate_epoch(test_dataloader, model, criterion, device, phase='test')
    print(f'Epoch {i}, train loss: {train_loss}, val loss: {val_loss}, val accuracy: {val_accuracy}')

test_loss, test_accuracy = evaluate_epoch(test_dataloader, model, criterion, device, phase='test')
print(f'Test loss: {test_loss}, test accuracy: {test_accuracy}')