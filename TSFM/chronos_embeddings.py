import torch
from tqdm import tqdm
import numpy as np
from chronos import ChronosPipeline
from classification_dataset import ClassificationDataset
from statistical_classifiers import fit_svm
from torch.utils.data import DataLoader
import argparse

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate embeddings using Chronos model')
    parser.add_argument('--output_file', type=str, default='./Chronos_DodgerLoopDay.txt',
                      help='Path to save the output predictions')
    parser.add_argument('--model_path', type=str, default='chronos-t5-large',
                      help='Path or name of the pretrained Chronos model')
    parser.add_argument('--train_file', type=str, default='DodgerLoopDay_TRAIN.ts',
                      help='Path to training data file')
    parser.add_argument('--test_file', type=str, default='DodgerLoopDay_TEST.ts',
                      help='Path to test data file')
    return parser.parse_args()

def get_embedding(model, dataloader):
    """Generate embeddings for all samples in the dataloader"""
    embeddings, labels = [], []
    with torch.no_grad():
        for batch_x, batch_masks, batch_labels in tqdm(dataloader, total=len(dataloader)):
            batch_x = batch_x.float()
            context = batch_x.squeeze()
            embedding, tokenizer_state = model.embed(context)
            embeddings.append(embedding.float().detach().cpu().numpy())
            labels.append(batch_labels)      

    embeddings, labels = np.concatenate(embeddings), np.concatenate(labels)
    return embeddings, labels

def main():
    args = parse_args()
    
    # Initialize model
    model = ChronosPipeline.from_pretrained(
        args.model_path,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )

    # Create datasets and dataloaders
    train_dataset = ClassificationDataset(
        data_split='train',
        train_file=args.train_file,
        test_file=args.test_file
    )
    test_dataset = ClassificationDataset(
        data_split='test',
        train_file=args.train_file,
        test_file=args.test_file
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)

    # Generate embeddings
    train_embeddings, train_labels = get_embedding(model, train_dataloader)
    test_embeddings, test_labels = get_embedding(model, test_dataloader)

    # Mean pooling
    train_embeddings = train_embeddings.mean(axis=1) 
    test_embeddings = test_embeddings.mean(axis=1)  

    print(f"Train embeddings shape: {train_embeddings.shape}, labels shape: {train_labels.shape}")
    print(f"Test embeddings shape: {test_embeddings.shape}, labels shape: {test_labels.shape}")

    # Train and evaluate SVM classifier
    clf = fit_svm(features=train_embeddings, y=train_labels)
    train_accuracy = clf.score(train_embeddings, train_labels)
    test_accuracy = clf.score(test_embeddings, test_labels)

    # Save predictions
    with open(args.output_file, "w") as f:
        f.write("Batch_idx, Predicted_Label, True_Label, Logits\n") 
        for batch_idx, (batch_embeddings, batch_labels) in enumerate(zip(test_embeddings, test_labels)):
            logits = clf.decision_function([batch_embeddings])
            predicted_labels = clf.predict([batch_embeddings])
            f.write(f"{batch_idx}, {predicted_labels[0]}, {batch_labels}, {logits[0].tolist()}\n")

    print(f"Predictions saved to {args.output_file}")
    print(f"Train accuracy: {train_accuracy:.8f}")
    print(f"Test accuracy: {test_accuracy:.8f}")

if __name__ == "__main__":
    main()