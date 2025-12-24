"""
Phase 7: Boilerplate vs. Material Risk Detection
SEC 10-K Risk Factor Intelligence

This module detects:
1. Year-over-year changes in risk disclosures
2. Boilerplate vs. substantive risk language
3. Industry-level similarity patterns

Enhanced with SBERT (Sentence-BERT) for semantic similarity:
- TF-IDF catches lexical overlap (same words)
- SBERT catches semantic overlap (same meaning, different words)
  e.g., "liquidity risk" ≈ "cash flow concerns"
"""

import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import pyarrow.parquet as pq
import pyarrow.compute as pc
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'


class BoilerplateDetector:
    """Detect boilerplate vs. material risk disclosures."""

    def __init__(self, max_paragraph_length: int = 5000, max_company_text: int = 50000):
        # For YoY comparison: truncate individual paragraphs
        self.max_paragraph_length = max_paragraph_length
        # For industry similarity: truncate aggregated company text
        self.max_company_text = max_company_text
        self.tfidf = None
        self.company_vectors = {}
        self.df = None
        self.df_full = None  # Full text for industry similarity

    def load_data(self, truncate: bool = True) -> pd.DataFrame:
        """Load risk factor data.

        Args:
            truncate: If True, truncate paragraphs (for YoY analysis).
                     If False, keep full text (for industry similarity).
        """
        print("Loading data...")
        filepath = DATA_DIR / 'risk_paragraphs.parquet'

        table = pq.read_table(filepath, columns=[
            'cik', 'filing_year', 'risk_content', 'primary_category'
        ])

        if truncate:
            # Truncate for memory-efficient YoY comparison
            truncated = pc.utf8_slice_codeunits(
                table.column('risk_content'), 0, self.max_paragraph_length
            )
            table = table.set_column(
                table.schema.get_field_index('risk_content'),
                'risk_content',
                truncated
            )
            self.df = table.to_pandas()
        else:
            # Keep full text for industry similarity
            self.df_full = table.to_pandas()
            if self.df is None:
                self.df = self.df_full

        del table
        gc.collect()

        df_ref = self.df_full if self.df_full is not None else self.df
        print(f"  Loaded {len(df_ref):,} risk paragraphs")
        print(f"  Years: {df_ref['filing_year'].min()}-{df_ref['filing_year'].max()}")
        print(f"  Companies: {df_ref['cik'].nunique():,}")

        return df_ref

    def compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts using SequenceMatcher."""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def compute_tfidf_similarity(self, text1: str, text2: str) -> float:
        """Compute TF-IDF cosine similarity between two texts."""
        if self.tfidf is None:
            self.tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
            # Fit on both texts
            self.tfidf.fit([text1, text2])

        vectors = self.tfidf.transform([text1, text2])
        return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    def detect_year_over_year_changes(
        self,
        cik: str,
        year1: int,
        year2: int
    ) -> Dict:
        """
        Detect changes between two years for a company.

        Returns dict with:
        - similarity_score: Overall similarity (0-1)
        - new_risks: Risk paragraphs in year2 but not year1
        - removed_risks: Risk paragraphs in year1 but not year2
        - modified_risks: Similar but changed paragraphs
        """
        if self.df is None:
            self.load_data(truncate=True)  # Use truncated for efficiency

        # Get risk paragraphs for each year
        df1 = self.df[(self.df['cik'] == cik) & (self.df['filing_year'] == year1)]
        df2 = self.df[(self.df['cik'] == cik) & (self.df['filing_year'] == year2)]

        if len(df1) == 0 or len(df2) == 0:
            return {'error': f'No data for company {cik} in years {year1} and/or {year2}'}

        texts1 = df1['risk_content'].tolist()
        texts2 = df2['risk_content'].tolist()
        cats1 = df1['primary_category'].tolist()
        cats2 = df2['primary_category'].tolist()

        # Match paragraphs between years
        new_risks = []
        removed_risks = []
        modified_risks = []
        unchanged_risks = []

        # Track which paragraphs from year2 have been matched
        matched_year2 = set()

        for i, (t1, c1) in enumerate(zip(texts1, cats1)):
            best_match_idx = -1
            best_match_score = 0

            for j, (t2, c2) in enumerate(zip(texts2, cats2)):
                if j in matched_year2:
                    continue

                sim = self.compute_text_similarity(t1, t2)
                if sim > best_match_score:
                    best_match_score = sim
                    best_match_idx = j

            if best_match_score > 0.9:
                # Unchanged
                unchanged_risks.append({
                    'text': t1[:500],
                    'category': c1,
                    'similarity': best_match_score
                })
                matched_year2.add(best_match_idx)
            elif best_match_score > 0.5:
                # Modified
                modified_risks.append({
                    'old_text': t1[:300],
                    'new_text': texts2[best_match_idx][:300],
                    'category': c1,
                    'similarity': best_match_score
                })
                matched_year2.add(best_match_idx)
            else:
                # Removed (no good match in year2)
                removed_risks.append({
                    'text': t1[:500],
                    'category': c1
                })

        # New risks are those in year2 not matched
        for j, (t2, c2) in enumerate(zip(texts2, cats2)):
            if j not in matched_year2:
                new_risks.append({
                    'text': t2[:500],
                    'category': c2
                })

        # Overall similarity
        overall_similarity = len(unchanged_risks) / max(len(texts1), len(texts2))

        return {
            'company': cik,
            'year1': year1,
            'year2': year2,
            'total_risks_year1': len(texts1),
            'total_risks_year2': len(texts2),
            'unchanged_count': len(unchanged_risks),
            'modified_count': len(modified_risks),
            'new_count': len(new_risks),
            'removed_count': len(removed_risks),
            'overall_similarity': overall_similarity,
            'change_rate': 1 - overall_similarity,
            'new_risks': new_risks,
            'removed_risks': removed_risks,
            'modified_risks': modified_risks
        }

    def compute_industry_similarity(
        self,
        year: int,
        sample_size: int = 500
    ) -> pd.DataFrame:
        """
        Compute similarity of each company's risk disclosure to peers.
        High similarity = more boilerplate.

        Uses FULL text (not truncated) for meaningful comparison.
        """
        # Load full text if not already loaded
        if self.df_full is None:
            self.load_data(truncate=False)

        print(f"\nComputing industry similarity for {year}...")

        # Get data for the year - use full text
        year_df = self.df_full[self.df_full['filing_year'] == year].copy()

        # Aggregate risk content by company
        company_texts = year_df.groupby('cik')['risk_content'].apply(
            lambda x: ' '.join(x.astype(str))
        ).reset_index()

        # Truncate aggregated text for memory efficiency (but keep enough for analysis)
        company_texts['risk_content'] = company_texts['risk_content'].str[:self.max_company_text]

        if len(company_texts) > sample_size:
            company_texts = company_texts.sample(n=sample_size, random_state=42)

        print(f"  Analyzing {len(company_texts)} companies...")
        avg_text_len = company_texts['risk_content'].str.len().mean()
        print(f"  Avg text length: {avg_text_len:,.0f} chars")

        # Fit TF-IDF
        tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        vectors = tfidf.fit_transform(company_texts['risk_content'])

        # Compute pairwise similarities
        sim_matrix = cosine_similarity(vectors)

        # For each company, compute average similarity to others (excluding self)
        avg_similarities = []
        for i in range(len(sim_matrix)):
            mask = np.ones(len(sim_matrix), dtype=bool)
            mask[i] = False
            avg_sim = sim_matrix[i, mask].mean()
            avg_similarities.append(avg_sim)

        company_texts['avg_similarity'] = avg_similarities
        company_texts['boilerplate_score'] = company_texts['avg_similarity']

        # Recalibrated thresholds based on actual TF-IDF similarity distributions
        # Empirical testing shows mean ~0.29 with full text
        company_texts['disclosure_type'] = pd.cut(
            company_texts['boilerplate_score'],
            bins=[0, 0.20, 0.30, 0.40, 1.0],
            labels=['Highly Unique', 'Mostly Unique', 'Somewhat Generic', 'Highly Boilerplate']
        )

        # Print distribution info
        print(f"  Similarity range: {company_texts['boilerplate_score'].min():.3f} - {company_texts['boilerplate_score'].max():.3f}")
        print(f"  Mean similarity: {company_texts['boilerplate_score'].mean():.3f}")

        return company_texts[['cik', 'avg_similarity', 'boilerplate_score', 'disclosure_type']]

    def compute_semantic_similarity(
        self,
        year: int,
        sample_size: int = 200
    ) -> pd.DataFrame:
        """
        Compute SEMANTIC similarity using SBERT embeddings.

        Unlike TF-IDF (lexical), SBERT captures meaning:
        - "The company faces liquidity risk" ≈ "Firm has cash flow concerns"
        - These have low TF-IDF similarity but high semantic similarity

        This is more sophisticated than TF-IDF and catches paraphrased boilerplate.
        """
        # Load full text if not already loaded
        if self.df_full is None:
            self.load_data(truncate=False)

        print(f"\nComputing SEMANTIC similarity for {year} (SBERT)...")

        # Get data for the year
        year_df = self.df_full[self.df_full['filing_year'] == year].copy()

        # Aggregate by company
        company_texts = year_df.groupby('cik')['risk_content'].apply(
            lambda x: ' '.join(x.astype(str))
        ).reset_index()

        # Truncate for embedding efficiency (SBERT works well with ~512 tokens)
        company_texts['risk_content'] = company_texts['risk_content'].str[:5000]

        if len(company_texts) > sample_size:
            company_texts = company_texts.sample(n=sample_size, random_state=42)

        print(f"  Analyzing {len(company_texts)} companies...")
        print(f"  Loading SBERT model (all-MiniLM-L6-v2)...")

        # Load sentence transformer
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Generate embeddings
        print(f"  Generating embeddings...")
        embeddings = model.encode(
            company_texts['risk_content'].tolist(),
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Compute pairwise cosine similarities
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        sim_matrix = cos_sim(embeddings)

        # For each company, compute average similarity to others
        avg_similarities = []
        for i in range(len(sim_matrix)):
            mask = np.ones(len(sim_matrix), dtype=bool)
            mask[i] = False
            avg_sim = sim_matrix[i, mask].mean()
            avg_similarities.append(avg_sim)

        company_texts['semantic_similarity'] = avg_similarities

        # Thresholds for semantic similarity (typically higher than TF-IDF)
        # SBERT similarities tend to be higher because it captures meaning
        company_texts['semantic_disclosure_type'] = pd.cut(
            company_texts['semantic_similarity'],
            bins=[0, 0.40, 0.55, 0.70, 1.0],
            labels=['Highly Unique', 'Mostly Unique', 'Somewhat Generic', 'Highly Boilerplate']
        )

        print(f"  Semantic similarity range: {company_texts['semantic_similarity'].min():.3f} - {company_texts['semantic_similarity'].max():.3f}")
        print(f"  Mean semantic similarity: {company_texts['semantic_similarity'].mean():.3f}")

        # Clean up
        del model, embeddings
        gc.collect()

        return company_texts[['cik', 'semantic_similarity', 'semantic_disclosure_type']]

    def compare_similarity_methods(
        self,
        year: int,
        sample_size: int = 150
    ) -> pd.DataFrame:
        """
        Compare TF-IDF (lexical) vs SBERT (semantic) similarity.

        This analysis shows the value of semantic understanding:
        - Companies with high semantic but low lexical similarity
          are using different words for the same boilerplate concepts
        """
        print(f"\nComparing similarity methods for {year}...")

        # Get TF-IDF similarity
        tfidf_df = self.compute_industry_similarity(year, sample_size)

        # Get SBERT similarity
        sbert_df = self.compute_semantic_similarity(year, sample_size)

        # Merge on CIK
        comparison = tfidf_df.merge(sbert_df, on='cik', how='inner')

        # Compute difference
        comparison['semantic_minus_lexical'] = (
            comparison['semantic_similarity'] - comparison['boilerplate_score']
        )

        # Categorize the difference
        comparison['similarity_pattern'] = pd.cut(
            comparison['semantic_minus_lexical'],
            bins=[-1, -0.1, 0.1, 1],
            labels=['Lexical > Semantic', 'Similar', 'Semantic > Lexical']
        )

        print(f"\n  Comparison Results:")
        print(f"  Mean TF-IDF (lexical): {comparison['boilerplate_score'].mean():.3f}")
        print(f"  Mean SBERT (semantic): {comparison['semantic_similarity'].mean():.3f}")
        print(f"  Correlation: {comparison['boilerplate_score'].corr(comparison['semantic_similarity']):.3f}")

        return comparison

    def analyze_change_trends(
        self,
        sample_companies: int = 100
    ) -> pd.DataFrame:
        """Analyze year-over-year change trends across companies."""
        if self.df is None:
            self.load_data(truncate=True)  # Use truncated for YoY (memory efficient)

        print(f"\nAnalyzing change trends for {sample_companies} companies...")

        # Find companies with multiple consecutive years
        company_years = self.df.groupby('cik')['filing_year'].apply(set).reset_index()
        company_years['years_list'] = company_years['filing_year'].apply(sorted)
        company_years['num_years'] = company_years['filing_year'].apply(len)

        # Filter to companies with 3+ years
        good_companies = company_years[company_years['num_years'] >= 3]

        if len(good_companies) > sample_companies:
            good_companies = good_companies.sample(n=sample_companies, random_state=42)

        results = []
        for _, row in good_companies.iterrows():
            cik = row['cik']
            years = row['years_list']

            for i in range(len(years) - 1):
                year1, year2 = years[i], years[i + 1]
                if year2 - year1 == 1:  # Consecutive years only
                    change = self.detect_year_over_year_changes(cik, year1, year2)
                    if 'error' not in change:
                        results.append({
                            'cik': cik,
                            'year': year2,
                            'change_rate': change['change_rate'],
                            'new_risks': change['new_count'],
                            'removed_risks': change['removed_count'],
                            'modified_risks': change['modified_count']
                        })

        return pd.DataFrame(results)

    def generate_report(self, output_dir: Path = None, include_semantic: bool = True) -> Dict:
        """Generate comprehensive boilerplate analysis report."""
        if output_dir is None:
            output_dir = OUTPUTS_DIR
        output_dir.mkdir(exist_ok=True)

        print("=" * 60)
        print("BOILERPLATE DETECTION ANALYSIS")
        print("Enhanced with Semantic Similarity (SBERT)")
        print("=" * 60)

        # 1. Industry similarity analysis (TF-IDF - lexical)
        print("\n[1/4] Computing LEXICAL similarity (TF-IDF)...")
        similarity_df = self.compute_industry_similarity(year=2020, sample_size=200)

        # 2. Semantic similarity analysis (SBERT)
        comparison_df = None
        if include_semantic:
            print("\n[2/4] Computing SEMANTIC similarity (SBERT)...")
            semantic_df = self.compute_semantic_similarity(year=2020, sample_size=200)

            # Merge for comparison
            comparison_df = similarity_df.merge(semantic_df, on='cik', how='inner')
            comparison_df['semantic_minus_lexical'] = (
                comparison_df['semantic_similarity'] - comparison_df['boilerplate_score']
            )

        # 3. Change trend analysis
        print("\n[3/4] Analyzing year-over-year changes...")
        change_df = self.analyze_change_trends(sample_companies=50)

        # 4. Case study - specific company
        print("\n[4/4] Generating case study...")

        # Find a company with interesting changes
        if len(change_df) > 0:
            high_change = change_df.nlargest(5, 'change_rate')
            case_company = high_change.iloc[0]['cik']
            case_year = int(high_change.iloc[0]['year'])
            case_study = self.detect_year_over_year_changes(
                case_company, case_year - 1, case_year
            )
        else:
            case_study = None

        # Generate visualizations
        self._plot_similarity_distribution(similarity_df, output_dir)
        self._plot_change_trends(change_df, output_dir)
        if comparison_df is not None:
            self._plot_similarity_comparison(comparison_df, output_dir)

        # Summary statistics
        report = {
            'lexical_analysis': {
                'year': 2020,
                'method': 'TF-IDF',
                'companies_analyzed': len(similarity_df),
                'avg_similarity': similarity_df['boilerplate_score'].mean(),
                'disclosure_type_distribution': similarity_df['disclosure_type'].value_counts().to_dict()
            },
            'change_analysis': {
                'companies_analyzed': change_df['cik'].nunique() if len(change_df) > 0 else 0,
                'year_pairs_analyzed': len(change_df),
                'avg_change_rate': change_df['change_rate'].mean() if len(change_df) > 0 else 0,
                'avg_new_risks_per_year': change_df['new_risks'].mean() if len(change_df) > 0 else 0
            },
            'case_study': case_study
        }

        # Add semantic analysis if available
        if comparison_df is not None:
            report['semantic_analysis'] = {
                'year': 2020,
                'method': 'SBERT (all-MiniLM-L6-v2)',
                'companies_analyzed': len(comparison_df),
                'avg_similarity': comparison_df['semantic_similarity'].mean(),
                'disclosure_type_distribution': comparison_df['semantic_disclosure_type'].value_counts().to_dict()
            }
            report['comparison'] = {
                'correlation': comparison_df['boilerplate_score'].corr(comparison_df['semantic_similarity']),
                'avg_difference': comparison_df['semantic_minus_lexical'].mean()
            }

        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        print(f"\n📊 LEXICAL SIMILARITY (TF-IDF)")
        print(f"   Companies analyzed: {report['lexical_analysis']['companies_analyzed']}")
        print(f"   Avg similarity: {report['lexical_analysis']['avg_similarity']:.3f}")
        print(f"   Distribution:")
        for dtype, count in report['lexical_analysis']['disclosure_type_distribution'].items():
            print(f"     - {dtype}: {count}")

        if 'semantic_analysis' in report:
            print(f"\n🧠 SEMANTIC SIMILARITY (SBERT)")
            print(f"   Companies analyzed: {report['semantic_analysis']['companies_analyzed']}")
            print(f"   Avg similarity: {report['semantic_analysis']['avg_similarity']:.3f}")
            print(f"   Distribution:")
            for dtype, count in report['semantic_analysis']['disclosure_type_distribution'].items():
                print(f"     - {dtype}: {count}")

            print(f"\n🔍 COMPARISON: Lexical vs Semantic")
            print(f"   Correlation: {report['comparison']['correlation']:.3f}")
            print(f"   Avg difference (semantic - lexical): {report['comparison']['avg_difference']:.3f}")
            if report['comparison']['avg_difference'] > 0.1:
                print(f"   → Semantic similarity is higher: companies use different words")
                print(f"     for similar risk concepts (paraphrased boilerplate)")

        print(f"\n📈 CHANGE ANALYSIS")
        print(f"   Companies analyzed: {report['change_analysis']['companies_analyzed']}")
        print(f"   Avg change rate: {report['change_analysis']['avg_change_rate']:.1%}")
        print(f"   Avg new risks/year: {report['change_analysis']['avg_new_risks_per_year']:.1f}")

        if case_study and 'error' not in case_study:
            print(f"\n📋 CASE STUDY: Company {case_study['company']}")
            print(f"   Period: {case_study['year1']} → {case_study['year2']}")
            print(f"   Change rate: {case_study['change_rate']:.1%}")
            print(f"   New risks: {case_study['new_count']}")
            print(f"   Removed risks: {case_study['removed_count']}")
            print(f"   Modified risks: {case_study['modified_count']}")

            if case_study['new_risks']:
                print(f"\n   📌 Sample NEW risk disclosed:")
                new_risk = case_study['new_risks'][0]
                print(f"      Category: {new_risk['category']}")
                print(f"      Text: {new_risk['text'][:200]}...")

        return report

    def _plot_similarity_distribution(self, df: pd.DataFrame, output_dir: Path):
        """Plot boilerplate score distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        axes[0].hist(df['boilerplate_score'], bins=30, edgecolor='black', alpha=0.7)
        axes[0].axvline(df['boilerplate_score'].mean(), color='red', linestyle='--',
                       label=f'Mean: {df["boilerplate_score"].mean():.3f}')
        axes[0].set_xlabel('Boilerplate Score (Avg Similarity to Peers)')
        axes[0].set_ylabel('Number of Companies')
        axes[0].set_title('Distribution of Boilerplate Scores')
        axes[0].legend()

        # Pie chart of disclosure types
        type_counts = df['disclosure_type'].value_counts()
        axes[1].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
                   colors=['#2ecc71', '#3498db', '#f39c12', '#e74c3c'])
        axes[1].set_title('Disclosure Type Distribution')

        plt.tight_layout()
        plt.savefig(output_dir / 'boilerplate_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Saved: {output_dir / 'boilerplate_distribution.png'}")

    def _plot_change_trends(self, df: pd.DataFrame, output_dir: Path):
        """Plot year-over-year change trends."""
        if len(df) == 0:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Change rate by year
        yearly_change = df.groupby('year')['change_rate'].mean()
        axes[0].bar(yearly_change.index, yearly_change.values, color='steelblue', edgecolor='black')
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Average Change Rate')
        axes[0].set_title('Year-over-Year Risk Disclosure Change Rate')
        axes[0].set_ylim(0, 1)

        # New risks by year
        yearly_new = df.groupby('year')['new_risks'].mean()
        axes[1].bar(yearly_new.index, yearly_new.values, color='#27ae60', edgecolor='black')
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('Avg New Risk Factors')
        axes[1].set_title('New Risk Factors Added Per Year')

        plt.tight_layout()
        plt.savefig(output_dir / 'change_trends.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Saved: {output_dir / 'change_trends.png'}")

    def _plot_similarity_comparison(self, df: pd.DataFrame, output_dir: Path):
        """Plot comparison of lexical vs semantic similarity."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Scatter plot: Lexical vs Semantic
        axes[0].scatter(df['boilerplate_score'], df['semantic_similarity'],
                       alpha=0.6, edgecolor='black', linewidth=0.5)
        axes[0].plot([0, 1], [0, 1], 'r--', label='Equal similarity')
        axes[0].set_xlabel('Lexical Similarity (TF-IDF)')
        axes[0].set_ylabel('Semantic Similarity (SBERT)')
        axes[0].set_title('Lexical vs Semantic Similarity\n(Points above line = semantic > lexical)')
        axes[0].legend()
        axes[0].set_xlim(0, max(df['boilerplate_score'].max() + 0.05, 0.5))
        axes[0].set_ylim(0, max(df['semantic_similarity'].max() + 0.05, 0.8))

        # 2. Distribution comparison
        axes[1].hist(df['boilerplate_score'], bins=20, alpha=0.7, label='Lexical (TF-IDF)',
                    color='steelblue', edgecolor='black')
        axes[1].hist(df['semantic_similarity'], bins=20, alpha=0.7, label='Semantic (SBERT)',
                    color='coral', edgecolor='black')
        axes[1].axvline(df['boilerplate_score'].mean(), color='steelblue', linestyle='--', linewidth=2)
        axes[1].axvline(df['semantic_similarity'].mean(), color='coral', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Similarity Score')
        axes[1].set_ylabel('Number of Companies')
        axes[1].set_title('Distribution: Lexical vs Semantic Similarity')
        axes[1].legend()

        # 3. Difference distribution
        diff = df['semantic_similarity'] - df['boilerplate_score']
        colors = ['coral' if d > 0 else 'steelblue' for d in diff]
        axes[2].hist(diff, bins=20, color='purple', alpha=0.7, edgecolor='black')
        axes[2].axvline(0, color='black', linestyle='-', linewidth=2)
        axes[2].axvline(diff.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {diff.mean():.3f}')
        axes[2].set_xlabel('Semantic - Lexical Similarity')
        axes[2].set_ylabel('Number of Companies')
        axes[2].set_title('Difference: Semantic vs Lexical\n(Positive = semantic captures more similarity)')
        axes[2].legend()

        plt.tight_layout()
        plt.savefig(output_dir / 'similarity_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Saved: {output_dir / 'similarity_comparison.png'}")


def main():
    """Run boilerplate detection analysis."""
    # max_paragraph_length: for YoY comparison (individual paragraphs)
    # max_company_text: for industry similarity (aggregated company text)
    detector = BoilerplateDetector(max_paragraph_length=5000, max_company_text=50000)
    report = detector.generate_report(include_semantic=True)

    print("\n" + "=" * 60)
    print("BOILERPLATE DETECTION COMPLETE")
    print("=" * 60)
    print("Outputs saved to: outputs/")
    print("  - boilerplate_distribution.png (lexical similarity)")
    print("  - change_trends.png (year-over-year changes)")
    print("  - similarity_comparison.png (lexical vs semantic)")


if __name__ == '__main__':
    main()
