"""
Phase 3: Optimized TF-IDF Classifier
SEC 10-K Risk Factor Classification

Optimizations:
- XGBoost and LightGBM (gradient boosting)
- Hyperparameter tuning
- Voting ensemble of best models
- Optimized TF-IDF parameters
"""

import gc
import argparse
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow.compute as pc
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
from collections import Counter
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


def load_data(max_text_length: int = 5000) -> pd.DataFrame:
    """Load full parquet data using PyArrow with text truncation."""
    filepath = DATA_DIR / 'risk_paragraphs.parquet'
    print(f"Loading data from {filepath.name}...")

    print("  Reading with PyArrow...")
    table = pq.read_table(filepath, columns=['risk_content', 'primary_category'])
    print(f"  Loaded {table.num_rows:,} rows ({table.nbytes / 1e9:.2f} GB)")

    print(f"  Truncating text to {max_text_length:,} chars...")
    truncated = pc.utf8_slice_codeunits(table.column('risk_content'), 0, max_text_length)
    table = table.set_column(0, 'risk_content', truncated)

    df = table.to_pandas()
    del table
    gc.collect()

    df = df[df['primary_category'] != 'other'].copy()
    df['primary_category'] = df['primary_category'].astype(str)

    print(f"  Final: {len(df):,} rows, {df.memory_usage(deep=True).sum() / 1e6:.0f} MB")
    return df


def balanced_sample(df: pd.DataFrame, min_samples: int = 500, max_samples: int = 5000) -> pd.DataFrame:
    """Balance dataset via up/downsampling."""
    print("\nBalancing dataset...")
    class_counts = df['primary_category'].value_counts()

    balanced_parts = []
    for category, count in class_counts.items():
        cat_df = df[df['primary_category'] == category]

        if count < min_samples:
            sampled = cat_df.sample(n=min_samples, replace=True, random_state=42)
            print(f"  {category}: {count} -> {min_samples} (upsampled)")
        elif count > max_samples:
            sampled = cat_df.sample(n=max_samples, random_state=42)
            print(f"  {category}: {count} -> {max_samples} (downsampled)")
        else:
            sampled = cat_df
            print(f"  {category}: {count} (kept)")

        balanced_parts.append(sampled)

    balanced_df = pd.concat(balanced_parts, ignore_index=True)
    print(f"\n  Balanced dataset: {len(balanced_df):,} samples")
    return balanced_df


def create_tfidf_features(X_train, X_test):
    """Create optimized TF-IDF features."""
    print("\n1. TF-IDF Vectorization (optimized)...")
    tfidf = TfidfVectorizer(
        max_features=10000,       # Back to 10K (15K was too slow)
        ngram_range=(1, 2),       # Bigrams only (trigrams too slow)
        min_df=5,
        max_df=0.95,
        stop_words='english',
        sublinear_tf=True,
        dtype=np.float32
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    print(f"   Shape: {X_train_tfidf.shape}")
    print(f"   Vocabulary: {len(tfidf.vocabulary_):,} terms")

    return tfidf, X_train_tfidf, X_test_tfidf


def train_all_models(X_train, y_train, X_test, y_test, le):
    """Train all models including gradient boosting."""
    print("\n" + "=" * 60)
    print("TRAINING ALL MODELS")
    print("=" * 60)

    # Encode labels for XGBoost/LightGBM
    y_train_encoded = le.transform(y_train)
    y_test_encoded = le.transform(y_test)

    models = {}
    results = {}

    # 1. Logistic Regression
    print("\n[1/5] Logistic Regression...")
    lr = LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=42,
        n_jobs=1, solver='saga', C=1.0
    )
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr

    # 2. Linear SVM
    print("[2/5] Linear SVM...")
    svm = CalibratedClassifierCV(
        LinearSVC(class_weight='balanced', random_state=42, max_iter=2000, C=0.5),
        cv=3
    )
    svm.fit(X_train, y_train)
    models['Linear SVM'] = svm

    # 3. Random Forest (tuned)
    print("[3/5] Random Forest (tuned)...")
    rf = RandomForestClassifier(
        n_estimators=200,         # Reduced from 300
        max_depth=40,             # Reduced from 60
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=2                  # Allow some parallelism
    )
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf

    # 4. XGBoost (optimized for speed)
    print("[4/5] XGBoost...")
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100,         # Reduced from 300
        max_depth=6,              # Reduced from 8
        learning_rate=0.15,       # Faster learning
        subsample=0.8,
        colsample_bytree=0.6,     # Use fewer features per tree
        random_state=42,
        n_jobs=2,                 # Allow some parallelism
        tree_method='hist',       # Much faster for sparse data
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    xgb_clf.fit(X_train, y_train_encoded)
    models['XGBoost'] = xgb_clf

    # 5. LightGBM (optimized for speed)
    print("[5/5] LightGBM...")
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=100,         # Reduced from 300
        max_depth=8,              # Reduced from 10
        learning_rate=0.15,
        subsample=0.8,
        colsample_bytree=0.6,
        class_weight='balanced',
        random_state=42,
        n_jobs=2,
        verbose=-1
    )
    lgb_clf.fit(X_train, y_train_encoded)
    models['LightGBM'] = lgb_clf

    # Evaluate all models
    print("\n" + "=" * 60)
    print("MODEL RESULTS")
    print("=" * 60)

    for name, model in models.items():
        if name in ['XGBoost', 'LightGBM']:
            y_pred_encoded = model.predict(X_test)
            y_pred = le.inverse_transform(y_pred_encoded)
        else:
            y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_mac = f1_score(y_test, y_pred, average='macro')
        f1_wt = f1_score(y_test, y_pred, average='weighted')

        results[name] = {
            'accuracy': acc,
            'f1_macro': f1_mac,
            'f1_weighted': f1_wt,
            'model': model,
            'predictions': y_pred
        }

        print(f"\n{name}:")
        print(f"   Accuracy: {acc:.3f} | F1 macro: {f1_mac:.3f} | F1 weighted: {f1_wt:.3f}")

    return models, results


def create_ensemble(models, results, X_train, y_train, X_test, y_test, le):
    """Create voting ensemble of top 3 models."""
    print("\n" + "=" * 60)
    print("CREATING ENSEMBLE")
    print("=" * 60)

    # Sort by F1 macro and pick top 3
    sorted_models = sorted(results.items(), key=lambda x: x[1]['f1_macro'], reverse=True)
    top_3 = sorted_models[:3]

    print(f"\nTop 3 models for ensemble:")
    for name, res in top_3:
        print(f"  {name}: F1 macro = {res['f1_macro']:.3f}")

    # Create ensemble with sklearn-compatible models only
    ensemble_estimators = []
    for name, res in top_3:
        model = res['model']
        # Skip XGBoost/LightGBM for sklearn VotingClassifier (they need label encoding)
        if name not in ['XGBoost', 'LightGBM']:
            ensemble_estimators.append((name.replace(' ', '_'), model))

    if len(ensemble_estimators) >= 2:
        print(f"\nCreating voting ensemble with: {[e[0] for e in ensemble_estimators]}")
        ensemble = VotingClassifier(estimators=ensemble_estimators, voting='soft')
        ensemble.fit(X_train, y_train)

        y_pred_ens = ensemble.predict(X_test)
        acc = accuracy_score(y_test, y_pred_ens)
        f1_mac = f1_score(y_test, y_pred_ens, average='macro')
        f1_wt = f1_score(y_test, y_pred_ens, average='weighted')

        results['Voting Ensemble'] = {
            'accuracy': acc,
            'f1_macro': f1_mac,
            'f1_weighted': f1_wt,
            'model': ensemble,
            'predictions': y_pred_ens
        }

        print(f"\nVoting Ensemble:")
        print(f"   Accuracy: {acc:.3f} | F1 macro: {f1_mac:.3f} | F1 weighted: {f1_wt:.3f}")

    # Also try simple averaging of XGBoost + LightGBM + RF predictions
    print("\nCreating soft voting ensemble (all top models)...")

    # Get probabilities from each model
    prob_sum = None
    prob_count = 0

    for name, res in sorted_models[:4]:  # Top 4 models
        model = res['model']
        if hasattr(model, 'predict_proba'):
            if name in ['XGBoost', 'LightGBM']:
                probs = model.predict_proba(X_test)
            else:
                probs = model.predict_proba(X_test)

            if prob_sum is None:
                prob_sum = probs
            else:
                # Align probabilities if needed
                if probs.shape == prob_sum.shape:
                    prob_sum += probs
                    prob_count += 1

    if prob_sum is not None and prob_count > 0:
        prob_avg = prob_sum / (prob_count + 1)
        y_pred_avg = le.inverse_transform(np.argmax(prob_avg, axis=1))

        acc = accuracy_score(y_test, y_pred_avg)
        f1_mac = f1_score(y_test, y_pred_avg, average='macro')
        f1_wt = f1_score(y_test, y_pred_avg, average='weighted')

        results['Soft Voting (All)'] = {
            'accuracy': acc,
            'f1_macro': f1_mac,
            'f1_weighted': f1_wt,
            'model': None,  # Can't save this easily
            'predictions': y_pred_avg
        }

        print(f"\nSoft Voting (All):")
        print(f"   Accuracy: {acc:.3f} | F1 macro: {f1_mac:.3f} | F1 weighted: {f1_wt:.3f}")

    return results


def detailed_evaluation(y_test, y_pred, model_name):
    """Print detailed evaluation."""
    print(f"\n{'=' * 60}")
    print(f"DETAILED EVALUATION: {model_name}")
    print(f"{'=' * 60}")

    print("\nClassification Report:")
    print("-" * 60)
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\nPer-Class Analysis:")
    print("-" * 60)

    labels = sorted(set(y_test))
    for label in labels:
        mask = np.array(y_test) == label
        total = mask.sum()
        pred_mask = np.array(y_pred) == label
        correct = (mask & pred_mask).sum()

        confused_with = Counter(np.array(y_pred)[mask])
        if label in confused_with:
            del confused_with[label]
        top_confusion = confused_with.most_common(2)

        print(f"\n{label}:")
        print(f"  Correct: {correct}/{total} ({100*correct/total:.1f}%)")
        if top_confusion:
            print(f"  Confused with: {', '.join([f'{k} ({v})' for k, v in top_confusion])}")


def plot_results(y_test, y_pred, results, model_name):
    """Generate all plots."""
    OUTPUTS_DIR.mkdir(exist_ok=True)

    # Confusion matrix
    labels = sorted(set(y_test) | set(y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_title(f'{model_name} - Counts')
    axes[0].tick_params(axis='x', rotation=45)

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].set_title(f'{model_name} - Recall')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / 'confusion_matrix_optimized.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nConfusion matrix saved: {OUTPUTS_DIR / 'confusion_matrix_optimized.png'}")

    # Model comparison
    model_names = [n for n in results.keys() if results[n]['model'] is not None or 'Soft' in n]
    metrics = ['accuracy', 'f1_macro', 'f1_weighted']

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(model_names))
    width = 0.25

    for i, metric in enumerate(metrics):
        values = [results[m][metric] for m in model_names]
        bars = ax.bar(x + i*width, values, width, label=metric.replace('_', ' ').title())

    ax.set_ylabel('Score')
    ax.set_title('Model Comparison (Optimized)')
    ax.set_xticks(x + width)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / 'model_comparison_optimized.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Model comparison saved: {OUTPUTS_DIR / 'model_comparison_optimized.png'}")


def save_artifacts(tfidf, best_model, best_name, results, le):
    """Save best model and metrics."""
    MODELS_DIR.mkdir(exist_ok=True)

    if best_model is not None:
        joblib.dump(best_model, MODELS_DIR / 'best_model_optimized.pkl')
        print(f"\nBest model saved: {MODELS_DIR / 'best_model_optimized.pkl'}")

    joblib.dump(tfidf, MODELS_DIR / 'tfidf_vectorizer_optimized.pkl')
    joblib.dump(le, MODELS_DIR / 'label_encoder.pkl')

    metrics_data = []
    for name, res in results.items():
        metrics_data.append({
            'model': name,
            'accuracy': res['accuracy'],
            'f1_macro': res['f1_macro'],
            'f1_weighted': res['f1_weighted'],
            'is_best': name == best_name
        })

    pd.DataFrame(metrics_data).to_csv(MODELS_DIR / 'model_comparison_optimized.csv', index=False)
    print(f"Metrics saved: {MODELS_DIR / 'model_comparison_optimized.csv'}")


def main():
    parser = argparse.ArgumentParser(description='Train optimized TF-IDF classifier')
    parser.add_argument('--max-text-length', type=int, default=5000)
    args = parser.parse_args()

    print("=" * 60)
    print("SEC RISK FACTOR CLASSIFIER - OPTIMIZED")
    print("XGBoost + LightGBM + Ensemble")
    print("=" * 60)

    # Load and balance data
    df = load_data(max_text_length=args.max_text_length)
    df = balanced_sample(df, min_samples=500, max_samples=5000)

    X = df['risk_content'].values
    y = df['primary_category'].values

    # Encode labels
    le = LabelEncoder()
    le.fit(y)

    del df
    gc.collect()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain/Test Split: {len(X_train):,} / {len(X_test):,}")

    # Create features
    tfidf, X_train_tfidf, X_test_tfidf = create_tfidf_features(X_train, X_test)

    # Train all models
    models, results = train_all_models(X_train_tfidf, y_train, X_test_tfidf, y_test, le)

    # Create ensemble
    results = create_ensemble(models, results, X_train_tfidf, y_train, X_test_tfidf, y_test, le)

    # Find best model
    best_name = max(results.keys(), key=lambda k: results[k]['f1_macro'])
    best_res = results[best_name]

    print(f"\n{'=' * 60}")
    print(f"BEST MODEL: {best_name}")
    print(f"   Accuracy:    {best_res['accuracy']:.3f}")
    print(f"   F1 (macro):  {best_res['f1_macro']:.3f}")
    print(f"   F1 (weighted): {best_res['f1_weighted']:.3f}")
    print(f"{'=' * 60}")

    # Detailed evaluation
    detailed_evaluation(y_test, best_res['predictions'], best_name)

    # Plots
    plot_results(y_test, best_res['predictions'], results, best_name)

    # Save
    save_artifacts(tfidf, best_res.get('model'), best_name, results, le)

    # Final summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Best Model:       {best_name}")
    print(f"Accuracy:         {best_res['accuracy']:.3f}")
    print(f"F1 (macro):       {best_res['f1_macro']:.3f}")
    print(f"F1 (weighted):    {best_res['f1_weighted']:.3f}")


if __name__ == '__main__':
    main()
