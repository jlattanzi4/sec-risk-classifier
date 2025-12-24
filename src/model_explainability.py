"""
Model Explainability with SHAP
SEC 10-K Risk Factor Classification

Explains model predictions using SHAP (SHapley Additive exPlanations).
Critical for regulated industries where decisions must be interpretable.

This demonstrates:
- Feature importance analysis
- Per-prediction explanations
- Understanding what drives each risk category
"""

import gc
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import pyarrow.parquet as pq
import pyarrow.compute as pc
import pandas as pd
import numpy as np
import shap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
MODELS_DIR = PROJECT_ROOT / 'models'


def load_data(max_text_length: int = 3000, sample_size: int = 5000) -> pd.DataFrame:
    """Load risk factor data."""
    print("Loading data...")
    filepath = DATA_DIR / 'risk_paragraphs.parquet'

    table = pq.read_table(filepath, columns=['risk_content', 'primary_category'])

    # Truncate text
    truncated = pc.utf8_slice_codeunits(table.column('risk_content'), 0, max_text_length)
    table = table.set_column(0, 'risk_content', truncated)

    df = table.to_pandas()
    del table
    gc.collect()

    # Filter out 'other'
    df = df[df['primary_category'] != 'other'].copy()

    # Sample for SHAP efficiency (SHAP can be slow on large datasets)
    if len(df) > sample_size:
        df = df.groupby('primary_category', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), sample_size // df['primary_category'].nunique()),
                             random_state=42)
        )

    print(f"  Loaded {len(df):,} paragraphs")
    return df.reset_index(drop=True)


def train_interpretable_model(
    df: pd.DataFrame
) -> Tuple[LogisticRegression, TfidfVectorizer, LabelEncoder, np.ndarray, np.ndarray]:
    """Train a logistic regression model for SHAP analysis.

    We use LogisticRegression because:
    1. Linear models have exact SHAP values (fast computation)
    2. Coefficients are interpretable
    3. Still performs well on text classification
    """
    print("\nTraining interpretable model...")

    # Encode labels
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['primary_category'])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['risk_content'].values,
        df['label'].values,
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )

    # TF-IDF (smaller vocabulary for interpretability)
    print("  Fitting TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(
        max_features=2000,  # Limited for clearer explanations
        stop_words='english',
        ngram_range=(1, 2),
        min_df=5
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Train logistic regression
    print("  Training Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_tfidf, y_train)

    # Evaluate
    train_acc = model.score(X_train_tfidf, y_train)
    test_acc = model.score(X_test_tfidf, y_test)
    print(f"  Train accuracy: {train_acc:.3f}")
    print(f"  Test accuracy: {test_acc:.3f}")

    return model, tfidf, le, X_test, y_test


def compute_shap_values(
    model: LogisticRegression,
    tfidf: TfidfVectorizer,
    X_test: np.ndarray,
    n_samples: int = 500
) -> Tuple[shap.Explainer, List[np.ndarray], np.ndarray]:
    """Compute SHAP values for test samples."""
    print("\nComputing SHAP values...")
    print(f"  Analyzing {n_samples} samples...")

    # Sample for efficiency
    if len(X_test) > n_samples:
        indices = np.random.RandomState(42).choice(len(X_test), n_samples, replace=False)
        X_sample = X_test[indices]
    else:
        X_sample = X_test

    # Transform to TF-IDF
    X_tfidf = tfidf.transform(X_sample)

    # Create SHAP explainer for linear model
    explainer = shap.LinearExplainer(model, X_tfidf, feature_names=tfidf.get_feature_names_out())

    # Compute SHAP values
    shap_values = explainer.shap_values(X_tfidf)

    # Handle different SHAP output formats
    if isinstance(shap_values, np.ndarray):
        # Shape is (n_samples, n_features, n_classes) - need to transpose
        if len(shap_values.shape) == 3:
            # Convert to list of arrays per class
            shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
        elif len(shap_values.shape) == 2:
            # Binary or single output - wrap in list
            shap_values = [shap_values]

    print(f"  SHAP values: {len(shap_values)} classes, shape per class: {shap_values[0].shape}")

    return explainer, shap_values, X_tfidf


def analyze_feature_importance(
    shap_values: List[np.ndarray],
    feature_names: np.ndarray,
    class_names: List[str]
) -> pd.DataFrame:
    """Analyze global feature importance per class."""
    print("\nAnalyzing feature importance...")

    importance_data = []

    for class_idx, class_name in enumerate(class_names):
        # Get SHAP values for this class
        class_shap = shap_values[class_idx]

        # Handle sparse matrices
        if hasattr(class_shap, 'toarray'):
            class_shap = class_shap.toarray()

        # Mean absolute SHAP value per feature for this class
        mean_abs_shap = np.abs(class_shap).mean(axis=0)

        # Flatten if needed
        if len(mean_abs_shap.shape) > 1:
            mean_abs_shap = mean_abs_shap.flatten()

        # Top features for this class
        top_indices = np.argsort(mean_abs_shap)[-10:][::-1]

        for rank, idx in enumerate(top_indices):
            importance_data.append({
                'class': class_name,
                'feature': feature_names[idx],
                'importance': float(mean_abs_shap[idx]),
                'rank': rank + 1
            })

    return pd.DataFrame(importance_data)


def plot_explanations(
    shap_values: np.ndarray,
    X_tfidf: np.ndarray,
    feature_names: np.ndarray,
    class_names: List[str],
    importance_df: pd.DataFrame,
    output_dir: Path
):
    """Generate SHAP visualizations."""
    output_dir.mkdir(exist_ok=True)
    print("\nGenerating visualizations...")

    # 1. Global feature importance per class
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, class_name in enumerate(class_names[:6]):  # Top 6 classes
        class_data = importance_df[importance_df['class'] == class_name].head(10)

        ax = axes[idx]
        bars = ax.barh(range(len(class_data)), class_data['importance'].values)
        ax.set_yticks(range(len(class_data)))
        ax.set_yticklabels(class_data['feature'].values, fontsize=9)
        ax.set_xlabel('Mean |SHAP value|')
        ax.set_title(f'{class_name}', fontsize=11, fontweight='bold')
        ax.invert_yaxis()

    # Hide empty subplots if fewer than 6 classes
    for idx in range(len(class_names), 6):
        axes[idx].set_visible(False)

    plt.suptitle('Top Features Driving Each Risk Category\n(Higher SHAP = stronger influence on prediction)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_feature_importance_by_class.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: shap_feature_importance_by_class.png")

    # 2. Overall feature importance (across all classes)
    overall_importance = np.zeros(len(feature_names))
    for class_idx in range(len(class_names)):
        class_shap = shap_values[class_idx]
        if hasattr(class_shap, 'toarray'):
            class_shap = class_shap.toarray()
        mean_abs = np.abs(class_shap).mean(axis=0)
        if len(mean_abs.shape) > 1:
            mean_abs = mean_abs.flatten()
        overall_importance += mean_abs
    overall_importance /= len(class_names)

    top_overall = np.argsort(overall_importance)[-20:][::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top_overall)), overall_importance[top_overall], color='steelblue')
    ax.set_yticks(range(len(top_overall)))
    ax.set_yticklabels([feature_names[i] for i in top_overall], fontsize=10)
    ax.set_xlabel('Mean |SHAP value| (averaged across classes)')
    ax.set_title('Top 20 Most Important Features Overall\n(Words that matter most for risk classification)',
                 fontsize=12, fontweight='bold')
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_dir / 'shap_overall_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: shap_overall_importance.png")

    # 3. SHAP summary plot (beeswarm) for one class
    # Pick the class with most samples
    try:
        class_shap = shap_values[0]
        if hasattr(class_shap, 'toarray'):
            class_shap = class_shap.toarray()
        X_dense = X_tfidf.toarray() if hasattr(X_tfidf, 'toarray') else X_tfidf

        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(
            class_shap,
            X_dense,
            feature_names=feature_names,
            max_display=15,
            show=False
        )
        plt.title(f'SHAP Summary for "{class_names[0]}" Classification', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / 'shap_summary_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: shap_summary_plot.png")
    except Exception as e:
        print(f"  Could not generate summary plot: {e}")

    # 4. Feature importance comparison across categories
    pivot_df = importance_df.pivot(index='feature', columns='class', values='importance')
    pivot_df = pivot_df.fillna(0)

    # Get features that appear in top 10 for any class
    top_features = importance_df.groupby('feature')['importance'].max().nlargest(15).index

    fig, ax = plt.subplots(figsize=(14, 8))
    pivot_subset = pivot_df.loc[pivot_df.index.isin(top_features)]
    sns_plot = sns.heatmap(pivot_subset, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
    ax.set_title('Feature Importance Across Risk Categories\n(Darker = more important for that category)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Risk Category')
    ax.set_ylabel('Feature (word/phrase)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_importance_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: shap_importance_heatmap.png")


def explain_sample_predictions(
    model: LogisticRegression,
    tfidf: TfidfVectorizer,
    le: LabelEncoder,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_examples: int = 3
) -> List[Dict]:
    """Generate explanations for individual predictions."""
    print(f"\nGenerating {n_examples} example explanations...")

    feature_names = tfidf.get_feature_names_out()
    class_names = le.classes_

    examples = []

    for i in range(min(n_examples, len(X_test))):
        text = X_test[i]
        true_label = le.inverse_transform([y_test[i]])[0]

        # Get prediction
        X_tfidf = tfidf.transform([text])
        pred_proba = model.predict_proba(X_tfidf)[0]
        pred_label = le.inverse_transform([model.predict(X_tfidf)[0]])[0]

        # Get feature contributions
        coefs = model.coef_[y_test[i]]
        feature_values = X_tfidf.toarray()[0]

        # Contribution = coefficient * feature value
        contributions = coefs * feature_values
        nonzero_mask = feature_values > 0

        # Top positive and negative contributors
        nonzero_contribs = contributions[nonzero_mask]
        nonzero_features = feature_names[nonzero_mask]

        top_positive_idx = np.argsort(nonzero_contribs)[-3:][::-1]
        top_negative_idx = np.argsort(nonzero_contribs)[:3]

        examples.append({
            'text_preview': text[:200] + '...',
            'true_label': true_label,
            'predicted_label': pred_label,
            'confidence': pred_proba.max(),
            'top_positive_features': [(nonzero_features[j], nonzero_contribs[j])
                                      for j in top_positive_idx if nonzero_contribs[j] > 0],
            'top_negative_features': [(nonzero_features[j], nonzero_contribs[j])
                                      for j in top_negative_idx if nonzero_contribs[j] < 0]
        })

    return examples


def main():
    """Run explainability analysis."""
    print("=" * 60)
    print("MODEL EXPLAINABILITY WITH SHAP")
    print("SEC Risk Factor Classification")
    print("=" * 60)

    # Load data
    df = load_data(max_text_length=3000, sample_size=5000)

    # Train model
    model, tfidf, le, X_test, y_test = train_interpretable_model(df)

    feature_names = tfidf.get_feature_names_out()
    class_names = le.classes_.tolist()

    # Compute SHAP values
    explainer, shap_values, X_tfidf = compute_shap_values(model, tfidf, X_test, n_samples=500)

    # Analyze importance
    importance_df = analyze_feature_importance(shap_values, feature_names, class_names)

    # Plot
    plot_explanations(shap_values, X_tfidf, feature_names, class_names, importance_df, OUTPUTS_DIR)

    # Example explanations
    examples = explain_sample_predictions(model, tfidf, le, X_test, y_test, n_examples=3)

    # Save importance data
    importance_df.to_csv(OUTPUTS_DIR / 'shap_feature_importance.csv', index=False)
    print(f"\nSaved: shap_feature_importance.csv")

    # Print summary
    print("\n" + "=" * 60)
    print("EXPLAINABILITY SUMMARY")
    print("=" * 60)

    print("\nTop Features by Risk Category:")
    print("-" * 60)
    for class_name in class_names[:6]:
        class_features = importance_df[importance_df['class'] == class_name].head(5)
        features_str = ', '.join(class_features['feature'].tolist())
        print(f"  {class_name}:")
        print(f"    {features_str}")

    print("\n\nExample Prediction Explanation:")
    print("-" * 60)
    if examples:
        ex = examples[0]
        print(f"  Text: \"{ex['text_preview']}\"")
        print(f"  True label: {ex['true_label']}")
        print(f"  Predicted: {ex['predicted_label']} ({ex['confidence']:.1%} confidence)")
        print(f"  Key words pushing toward this category:")
        for feat, val in ex['top_positive_features'][:3]:
            print(f"    + {feat}: {val:.4f}")

    print("\n" + "=" * 60)
    print("EXPLAINABILITY ANALYSIS COMPLETE")
    print("=" * 60)
    print("Outputs:")
    print("  - shap_feature_importance_by_class.png")
    print("  - shap_overall_importance.png")
    print("  - shap_summary_plot.png")
    print("  - shap_importance_heatmap.png")
    print("  - shap_feature_importance.csv")


if __name__ == '__main__':
    main()
