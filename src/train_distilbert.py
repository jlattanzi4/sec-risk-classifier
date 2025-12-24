"""
Phase 4: DistilBERT Classifier
SEC 10-K Risk Factor Classification

Uses DistilBERT for semantic understanding of risk factors.
Optimized for Apple Silicon (MPS) with memory safeguards.
"""

import gc
import os
import argparse
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow.compute as pc
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'


def get_device():
    """Get the best available device."""
    if torch.backends.mps.is_available():
        print("Using Apple MPS (GPU)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA (GPU)")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")


class RiskDataset(Dataset):
    """PyTorch Dataset for risk factor texts."""

    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


def load_data(max_text_length: int = 2000) -> pd.DataFrame:
    """Load and prepare data."""
    filepath = DATA_DIR / 'risk_paragraphs.parquet'
    print(f"Loading data from {filepath.name}...")

    table = pq.read_table(filepath, columns=['risk_content', 'primary_category'])
    print(f"  Loaded {table.num_rows:,} rows")

    # Truncate text (DistilBERT has 512 token limit, ~2000 chars is safe)
    truncated = pc.utf8_slice_codeunits(table.column('risk_content'), 0, max_text_length)
    table = table.set_column(0, 'risk_content', truncated)

    df = table.to_pandas()
    del table
    gc.collect()

    df = df[df['primary_category'] != 'other'].copy()
    df['primary_category'] = df['primary_category'].astype(str)

    print(f"  Final: {len(df):,} rows")
    return df


def balanced_sample(df: pd.DataFrame, min_samples: int = 300, max_samples: int = 1500) -> pd.DataFrame:
    """Balance dataset - smaller samples for BERT (memory)."""
    print("\nBalancing dataset...")
    class_counts = df['primary_category'].value_counts()

    balanced_parts = []
    for category, count in class_counts.items():
        cat_df = df[df['primary_category'] == category]

        if count < min_samples:
            sampled = cat_df.sample(n=min_samples, replace=True, random_state=42)
            print(f"  {category}: {count} -> {min_samples}")
        elif count > max_samples:
            sampled = cat_df.sample(n=max_samples, random_state=42)
            print(f"  {category}: {count} -> {max_samples}")
        else:
            sampled = cat_df
            print(f"  {category}: {count}")

        balanced_parts.append(sampled)

    balanced_df = pd.concat(balanced_parts, ignore_index=True)
    print(f"\n  Balanced: {len(balanced_df):,} samples")
    return balanced_df


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Progress update every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"    Batch {batch_idx + 1}/{len(dataloader)} | Loss: {loss.item():.4f}")

        # Clear cache frequently (memory management)
        if (batch_idx + 1) % 25 == 0:
            if device.type == 'mps':
                torch.mps.empty_cache()
            elif device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    return avg_loss, np.array(all_preds), np.array(all_labels)


def plot_results(y_true, y_pred, label_names, history):
    """Generate plots."""
    OUTPUTS_DIR.mkdir(exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names, ax=axes[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_title('DistilBERT - Counts')
    axes[0].tick_params(axis='x', rotation=45)

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names, ax=axes[1])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].set_title('DistilBERT - Recall')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / 'confusion_matrix_distilbert.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Training history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()

    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / 'training_history_distilbert.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nPlots saved to {OUTPUTS_DIR}")


def main():
    parser = argparse.ArgumentParser(description='Train DistilBERT classifier')
    parser.add_argument('--epochs', type=int, default=4, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--max-length', type=int, default=256, help='Max token length')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    args = parser.parse_args()

    print("=" * 60)
    print("SEC RISK FACTOR CLASSIFIER - DistilBERT")
    print("=" * 60)

    device = get_device()

    # Load data
    df = load_data(max_text_length=2000)
    df = balanced_sample(df, min_samples=400, max_samples=3000)

    # Encode labels
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['primary_category'])
    label_names = le.classes_

    print(f"\nClasses: {list(label_names)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['risk_content'].values,
        df['label'].values,
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )

    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    del df
    gc.collect()

    # Load tokenizer and model
    print("\nLoading DistilBERT...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=len(label_names)
    )
    model.to(device)

    # Create datasets
    print("Creating datasets...")
    train_dataset = RiskDataset(X_train, y_train, tokenizer, max_length=args.max_length)
    test_dataset = RiskDataset(X_test, y_test, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Training loop
    print(f"\n{'=' * 60}")
    print(f"TRAINING ({args.epochs} epochs, batch_size={args.batch_size})")
    print(f"{'=' * 60}")

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_f1 = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        # Evaluate
        val_loss, y_pred, y_true = evaluate(model, test_loader, device)
        val_acc = accuracy_score(y_true, y_pred)
        val_f1 = f1_score(y_true, y_pred, average='macro')

        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            MODELS_DIR.mkdir(exist_ok=True)
            torch.save(model.state_dict(), MODELS_DIR / 'distilbert_best.pt')
            print(f"  ✓ New best model saved (F1: {val_f1:.4f})")

        # Clear cache
        if device.type == 'mps':
            torch.mps.empty_cache()

    # Final evaluation
    print(f"\n{'=' * 60}")
    print("FINAL EVALUATION")
    print(f"{'=' * 60}")

    # Load best model
    model.load_state_dict(torch.load(MODELS_DIR / 'distilbert_best.pt'))
    _, y_pred, y_true = evaluate(model, test_loader, device)

    y_true_labels = le.inverse_transform(y_true)
    y_pred_labels = le.inverse_transform(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    print(f"\nAccuracy:      {accuracy:.3f}")
    print(f"F1 (macro):    {f1_macro:.3f}")
    print(f"F1 (weighted): {f1_weighted:.3f}")

    print("\nClassification Report:")
    print("-" * 60)
    print(classification_report(y_true_labels, y_pred_labels, zero_division=0))

    # Plots
    plot_results(y_true, y_pred, label_names, history)

    # Save label encoder
    import joblib
    joblib.dump(le, MODELS_DIR / 'label_encoder_distilbert.pkl')
    joblib.dump(tokenizer, MODELS_DIR / 'tokenizer_distilbert.pkl')

    # Summary
    print(f"\n{'=' * 60}")
    print("DISTILBERT TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Best F1 (macro):  {best_f1:.3f}")
    print(f"Final Accuracy:   {accuracy:.3f}")
    print(f"Model saved:      {MODELS_DIR / 'distilbert_best.pt'}")


if __name__ == '__main__':
    main()
