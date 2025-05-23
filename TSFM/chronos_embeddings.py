import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from chronos import ChronosPipeline
from classification_dataset import ClassificationDataset
from statistical_classifiers import fit_svm


def get_embeddings(model: ChronosPipeline, dataloader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate embeddings from Chronos model for given dataset.
    
    Args:
        model: Pre-trained Chronos model
        dataloader: DataLoader containing time series data
        
    Returns:
        Tuple of (embeddings, labels) as numpy arrays
    """
    embeddings, labels = [], []
    
    with torch.no_grad():
        for batch_x, _, batch_labels in tqdm(dataloader, desc="Generating embeddings"):
            context = batch_x.squeeze().float()
            embedding, _ = model.embed(context)
            embeddings.append(embedding.float().detach().cpu().numpy())
            labels.append(batch_labels)

    return np.concatenate(embeddings), np.concatenate(labels)


def main():
    parser = argparse.ArgumentParser(description='Generate Chronos embeddings and train SVM classifier')
    parser.add_argument('--model_path', required=True, help='Path to pre-trained Chronos model')
    parser.add_argument('--output_file', default='./output/Chronos_predictions.txt', 
                       help='Path to save predictions')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for embedding generation')
    parser.add_argument('--pooling', choices=['mean', 'first'], default='mean',
                       help='Embedding pooling strategy (mean or first token)')
    
    args = parser.parse_args()

    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChronosPipeline.from_pretrained(
        args.model_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )

    # Prepare datasets
    train_dataset = ClassificationDataset(data_split='train')
    test_dataset = ClassificationDataset(data_split='test')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Generate embeddings
    train_embeddings, train_labels = get_embeddings(model, train_dataloader)
    test_embeddings, test_labels = get_embeddings(model, test_dataloader)

    # Apply pooling
    if args.pooling == 'mean':
        train_embeddings = train_embeddings.mean(axis=1)
        test_embeddings = test_embeddings.mean(axis=1)
    else:  # first token
        train_embeddings = train_embeddings[:, 0, :]
        test_embeddings = test_embeddings[:, 0, :]

    # Train and evaluate SVM
    clf = fit_svm(features=train_embeddings, y=train_labels)
    train_accuracy = clf.score(train_embeddings, train_labels)
    test_accuracy = clf.score(test_embeddings, test_labels)

    # Save predictions
    with open(args.output_file, "w") as f:
        f.write("Batch_idx, Predicted_Label, True_Label, Logits\n")
        for batch_idx, (emb, label) in enumerate(zip(test_embeddings, test_labels)):
            logits = clf.decision_function([emb])
            pred = clf.predict([emb])[0]
            f.write(f"{batch_idx}, {pred}, {label}, {logits[0].tolist()}\n")

    print(f"\nResults saved to {args.output_file}")
    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()