from momentfm import MOMENTPipeline
from torch.utils.data import DataLoader
from momentfm.data.classification_dataset import ClassificationDataset
import torch
from torch import nn
from pprint import pprint
import torch
from tqdm import tqdm
import numpy as np
from momentfm.models.statistical_classifiers import fit_svm
import random
import os
import argparse

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fine-tune MOMENT model for classification')
    parser.add_argument('--num_class', type=int, required=True, help='Number of classification classes')
    parser.add_argument('--model_path', type=str, default="D:/Models/MOMENT/MOMENT-1-large", help='Path to pre-trained MOMENT model')
    parser.add_argument('--output_file', type=str, default='./MOMENT_Acoustic_window200.txt', help='Output file for evaluation results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-6, help='Initial learning rate')
    parser.add_argument('--max_lr', type=float, default=1e-4, help='Maximum learning rate for OneCycleLR scheduler')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (cuda or cpu)')
    parser.add_argument('--reduction', type=str, default='mean', help='Embedding reduction method (mean or concat)')
    return parser.parse_args()

def control_randomness(seed: int = 42):
    """Control randomness for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(model, device, train_dataloader, criterion, optimizer, scheduler, reduction='mean'):
    """Train the model for one epoch."""
    model.to(device)
    model.train()
    losses = []

    for batch_idx, (batch_x, batch_masks, batch_labels) in enumerate(train_dataloader):
        optimizer.zero_grad()
        batch_x = batch_x.to(device).float()
        batch_labels = batch_labels.to(device).long()

        # Mixed precision training for faster computation
        with torch.autocast(device_type=device, dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32):
            output = model(x_enc=batch_x, reduction=reduction)
            loss = criterion(output.logits, batch_labels)
        loss.backward()

        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    
    avg_loss = np.mean(losses)
    return avg_loss

def evaluate_epoch(dataloader, model, criterion, device, phase='val', reduction='mean', output_file='./evaluation_results.txt'):
    """Evaluate the model on a dataset."""
    model.eval()
    model.to(device)
    total_loss, total_correct = 0, 0
    idx = 0

    # Save evaluation results to file
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

                # Save predictions and ground truth
                predicted_labels = output.logits.argmax(dim=1).cpu().numpy()
                true_labels = batch_labels.cpu().numpy()
                logits = output.logits.cpu().numpy()

                for i in range(len(predicted_labels)):
                    f.write(f"{idx}, {predicted_labels[i]}, {true_labels[i]}, {logits[i].tolist()}\n")
                    idx += 1
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / len(dataloader.dataset)
    return avg_loss, accuracy

def main():
    args = parse_args()
    control_randomness(args.seed)

    # Initialize model with specified parameters
    model = MOMENTPipeline.from_pretrained(
        args.model_path, 
        model_kwargs={
            'task_name': 'classification',
            'n_channels': 12,  # Number of input channels
            'num_class': args.num_class,
            'freeze_encoder': False,  # Train the patch embedding layer
            'freeze_embedder': False,  # Train the transformer encoder
            'freeze_head': False,  # Train the linear classification head
            'enable_gradient_checkpointing': False,  # Disable for stability
            'reduction': args.reduction,  # Embedding reduction method
        },
        local_files_only=True,  # Load model from local path
    )

    model.init()
    # model.to(args.device).float()

    # Print model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Total parameters: {total_params/1e9:.2f}B")
    print(f"Total parameters: {total_params / 1e6:.2f}M")

    # Prepare datasets and dataloaders
    train_dataset = ClassificationDataset(data_split='train')
    test_dataset = ClassificationDataset(data_split='test')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Training setup
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=args.max_lr, 
        total_steps=args.epochs * len(train_dataloader)
    )

    # Training loop
    for i in tqdm(range(args.epochs)):
        train_loss = train_epoch(model, args.device, train_dataloader, criterion, optimizer, scheduler, args.reduction)
        val_loss, val_accuracy = evaluate_epoch(test_dataloader, model, criterion, args.device, phase='test', output_file=args.output_file)
        print(f'Epoch {i}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, val accuracy: {val_accuracy:.4f}')

    # Final evaluation
    test_loss, test_accuracy = evaluate_epoch(test_dataloader, model, criterion, args.device, phase='test', output_file=args.output_file)
    print(f'Test loss: {test_loss:.4f}, test accuracy: {test_accuracy:.4f}')

if __name__ == "__main__":
    main()    