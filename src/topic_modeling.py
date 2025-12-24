"""
Topic Modeling with BERTopic
SEC 10-K Risk Factor Intelligence

Unsupervised discovery of latent risk themes using:
- Sentence embeddings (all-MiniLM-L6-v2)
- UMAP dimensionality reduction
- HDBSCAN clustering
- c-TF-IDF topic representation

This demonstrates modern NLP beyond simple classification.
"""

import gc
import os
from pathlib import Path
from typing import Dict, List, Tuple

# Prevent OpenMP issues on macOS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
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


def load_data(sample_size: int = 10000, max_text_length: int = 1000) -> pd.DataFrame:
    """Load and sample risk factor data.

    Args:
        sample_size: Number of paragraphs to sample (BERTopic is memory-intensive)
        max_text_length: Truncate text for embedding efficiency
    """
    print("Loading data...")
    filepath = DATA_DIR / 'risk_paragraphs.parquet'

    table = pq.read_table(filepath, columns=[
        'cik', 'filing_year', 'risk_content', 'primary_category'
    ])
    df = table.to_pandas()
    del table
    gc.collect()

    print(f"  Total paragraphs: {len(df):,}")

    # Filter out 'other' category for cleaner analysis
    df = df[df['primary_category'] != 'other'].copy()

    # Truncate text (embeddings work best with ~256-512 tokens)
    df['risk_content'] = df['risk_content'].str[:max_text_length]

    # Remove very short texts
    df = df[df['risk_content'].str.len() > 100]

    # Sample for memory efficiency
    if len(df) > sample_size:
        # Stratified sample by category
        df = df.groupby('primary_category', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), sample_size // df['primary_category'].nunique()),
                             random_state=42)
        )

    print(f"  Sampled: {len(df):,} paragraphs")
    print(f"  Categories: {df['primary_category'].nunique()}")
    print(f"  Years: {df['filing_year'].min()}-{df['filing_year'].max()}")

    return df.reset_index(drop=True)


def train_topic_model(
    docs: List[str],
    n_topics: int = None,
    min_topic_size: int = 50
) -> BERTopic:
    """Train BERTopic model.

    Args:
        docs: List of document texts
        n_topics: Number of topics (None for automatic)
        min_topic_size: Minimum documents per topic
    """
    print("\nTraining BERTopic model...")
    print("  Loading sentence transformer...")

    # Use a lightweight but effective model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Custom vectorizer for financial/legal text
    vectorizer = CountVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=10000
    )

    # Configure UMAP with stable settings for macOS
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=42,
        low_memory=True,  # Important for stability
        n_jobs=1  # Single thread to avoid seg faults
    )

    # Configure HDBSCAN
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=10,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True,
        core_dist_n_jobs=1  # Single thread
    )

    # Initialize BERTopic with explicit models
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        nr_topics=n_topics,  # None = automatic
        verbose=True
    )

    print("  Fitting model (this may take a few minutes)...")
    topics, probs = topic_model.fit_transform(docs)

    # Get topic info
    topic_info = topic_model.get_topic_info()
    n_topics_found = len(topic_info) - 1  # Exclude outlier topic (-1)

    print(f"\n  Topics discovered: {n_topics_found}")
    print(f"  Outliers: {(np.array(topics) == -1).sum():,} documents")

    return topic_model, topics, probs


def analyze_topic_category_alignment(
    topic_model: BERTopic,
    topics: List[int],
    categories: List[str]
) -> pd.DataFrame:
    """Analyze how discovered topics align with manual categories."""
    print("\nAnalyzing topic-category alignment...")

    # Create cross-tabulation
    df = pd.DataFrame({
        'topic': topics,
        'category': categories
    })

    # Remove outliers for cleaner analysis
    df = df[df['topic'] != -1]

    # Cross-tabulation
    crosstab = pd.crosstab(df['topic'], df['category'], normalize='index')

    # For each topic, find dominant category
    topic_categories = []
    for topic_id in crosstab.index:
        dominant_cat = crosstab.loc[topic_id].idxmax()
        dominance = crosstab.loc[topic_id].max()
        topic_words = topic_model.get_topic(topic_id)[:5]
        topic_words_str = ', '.join([w[0] for w in topic_words])

        topic_categories.append({
            'topic_id': topic_id,
            'dominant_category': dominant_cat,
            'category_purity': dominance,
            'top_words': topic_words_str
        })

    alignment_df = pd.DataFrame(topic_categories)

    # Summary statistics
    avg_purity = alignment_df['category_purity'].mean()
    print(f"  Average category purity: {avg_purity:.1%}")
    print(f"  (Higher = topics align well with manual categories)")

    return alignment_df, crosstab


def analyze_temporal_trends(
    topics: List[int],
    years: List[int],
    topic_model: BERTopic
) -> pd.DataFrame:
    """Analyze how topic prevalence changes over time."""
    print("\nAnalyzing temporal trends...")

    df = pd.DataFrame({
        'topic': topics,
        'year': years
    })

    # Remove outliers
    df = df[df['topic'] != -1]

    # Topic prevalence by year
    yearly_topics = df.groupby(['year', 'topic']).size().unstack(fill_value=0)

    # Normalize by year
    yearly_topics_pct = yearly_topics.div(yearly_topics.sum(axis=1), axis=0)

    return yearly_topics_pct


def plot_results(
    topic_model: BERTopic,
    topics: List[int],
    alignment_df: pd.DataFrame,
    crosstab: pd.DataFrame,
    temporal_df: pd.DataFrame,
    output_dir: Path
):
    """Generate visualizations."""
    output_dir.mkdir(exist_ok=True)

    print("\nGenerating visualizations...")

    # 1. Topic-Category Heatmap
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(crosstab, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
    ax.set_xlabel('Manual Category')
    ax.set_ylabel('Discovered Topic')
    ax.set_title('Topic-Category Alignment\n(Row-normalized: proportion of topic in each category)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'topic_category_alignment.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: topic_category_alignment.png")

    # 2. Topic Words Summary
    fig, ax = plt.subplots(figsize=(12, max(6, len(alignment_df) * 0.4)))

    # Create horizontal bar chart of topic purity
    colors = plt.cm.RdYlGn(alignment_df['category_purity'])
    bars = ax.barh(range(len(alignment_df)), alignment_df['category_purity'], color=colors)

    ax.set_yticks(range(len(alignment_df)))
    ax.set_yticklabels([f"Topic {row['topic_id']}: {row['top_words'][:40]}..."
                        for _, row in alignment_df.iterrows()], fontsize=8)
    ax.set_xlabel('Category Purity (how well topic aligns with one category)')
    ax.set_title('Discovered Topics and Their Alignment with Manual Categories')
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    ax.legend()
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / 'topic_purity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: topic_purity.png")

    # 3. Temporal Trends (top 5 topics)
    if len(temporal_df.columns) > 0:
        # Get top 5 most common topics
        topic_sizes = temporal_df.sum().nlargest(5)
        top_topics = topic_sizes.index.tolist()

        fig, ax = plt.subplots(figsize=(12, 6))

        for topic_id in top_topics:
            if topic_id in temporal_df.columns:
                topic_words = topic_model.get_topic(topic_id)[:3]
                label = f"Topic {topic_id}: {', '.join([w[0] for w in topic_words])}"
                ax.plot(temporal_df.index, temporal_df[topic_id], marker='o', label=label, linewidth=2)

        ax.set_xlabel('Year')
        ax.set_ylabel('Topic Prevalence')
        ax.set_title('Risk Topic Evolution Over Time (Top 5 Topics)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'topic_temporal_trends.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: topic_temporal_trends.png")

    # 4. Save BERTopic's built-in visualizations
    try:
        # Topic hierarchy (if enough topics)
        topic_info = topic_model.get_topic_info()
        if len(topic_info) > 5:
            fig = topic_model.visualize_hierarchy()
            fig.write_html(str(output_dir / 'topic_hierarchy.html'))
            print(f"  Saved: topic_hierarchy.html")
    except Exception as e:
        print(f"  Could not generate hierarchy: {e}")

    try:
        # Intertopic distance map
        fig = topic_model.visualize_topics()
        fig.write_html(str(output_dir / 'topic_distance_map.html'))
        print(f"  Saved: topic_distance_map.html")
    except Exception as e:
        print(f"  Could not generate distance map: {e}")


def main():
    """Run topic modeling analysis."""
    print("=" * 60)
    print("SEC RISK FACTOR TOPIC MODELING")
    print("Unsupervised Discovery with BERTopic")
    print("=" * 60)

    # Load data (reduced sample for stability on macOS)
    df = load_data(sample_size=5000, max_text_length=800)

    docs = df['risk_content'].tolist()
    categories = df['primary_category'].tolist()
    years = df['filing_year'].tolist()

    # Train topic model
    topic_model, topics, probs = train_topic_model(
        docs,
        n_topics=None,  # Automatic
        min_topic_size=50  # Larger clusters for stability
    )

    # Analysis
    alignment_df, crosstab = analyze_topic_category_alignment(
        topic_model, topics, categories
    )

    temporal_df = analyze_temporal_trends(topics, years, topic_model)

    # Visualizations
    plot_results(
        topic_model, topics, alignment_df, crosstab, temporal_df, OUTPUTS_DIR
    )

    # Save model
    MODELS_DIR.mkdir(exist_ok=True)
    topic_model.save(str(MODELS_DIR / 'bertopic_model'))
    print(f"\nModel saved to: {MODELS_DIR / 'bertopic_model'}")

    # Print summary
    print("\n" + "=" * 60)
    print("TOPIC MODELING SUMMARY")
    print("=" * 60)

    print("\nTop 10 Discovered Topics:")
    print("-" * 60)
    topic_info = topic_model.get_topic_info()
    for _, row in topic_info.head(11).iterrows():
        if row['Topic'] == -1:
            continue
        topic_words = topic_model.get_topic(row['Topic'])[:5]
        words = ', '.join([w[0] for w in topic_words])
        print(f"  Topic {row['Topic']:2d} ({row['Count']:4d} docs): {words}")

    print(f"\nKey Insight:")
    high_purity = alignment_df[alignment_df['category_purity'] > 0.5]
    print(f"  {len(high_purity)}/{len(alignment_df)} topics ({len(high_purity)/len(alignment_df):.0%}) "
          f"align strongly with manual categories")
    print(f"  This validates the manual taxonomy while revealing sub-themes")

    print("\n" + "=" * 60)
    print("TOPIC MODELING COMPLETE")
    print("=" * 60)
    print("Outputs:")
    print("  - topic_category_alignment.png")
    print("  - topic_purity.png")
    print("  - topic_temporal_trends.png")
    print("  - topic_hierarchy.html (interactive)")
    print("  - topic_distance_map.html (interactive)")


if __name__ == '__main__':
    main()
