"""Text preprocessing utilities for SEC Risk Factor classification."""

import re


def clean_risk_text(text):
    """Clean and normalize risk factor text."""
    if not isinstance(text, str):
        return ''

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # Remove HTML entities
    text = re.sub(r'&nbsp;|&#160;|&amp;|&quot;|&lt;|&gt;', ' ', text)

    # Remove navigation text
    text = re.sub(r'back to (table of )?contents', '', text, flags=re.IGNORECASE)

    # Remove Item 1A header
    text = re.sub(r'^\s*item\s*1a\.?\s*:?\s*risk\s*factors\s*', '', text, flags=re.IGNORECASE)

    # Normalize bullet characters
    text = re.sub(r'[•●○◦▪▫◘►▸‣⁃]', '•', text)

    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove page numbers
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\b\d+\s*of\s*\d+\b', '', text)
    text = re.sub(r'table of contents', '', text, flags=re.IGNORECASE)

    return text.strip()


def extract_risk_paragraphs(text):
    """Extract individual risk paragraphs from Item 1A text."""
    if not text or len(text) < 100:
        return []

    risks = []
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    current_header = None
    current_content = []

    for para in paragraphs:
        is_header = False

        if len(para) < 200:
            if para.isupper():
                is_header = True
            elif para.endswith('.') and len(para) < 150:
                if re.match(r'^(we |our |the |if |there |risks? |loss |failure |changes? |inability )', para, re.I):
                    is_header = True
            elif para.istitle() and len(para) < 100:
                is_header = True

        if is_header:
            if current_header and current_content:
                risks.append({
                    'header': current_header,
                    'content': ' '.join(current_content)
                })
            current_header = para
            current_content = []
        else:
            current_content.append(para)

    if current_header and current_content:
        risks.append({
            'header': current_header,
            'content': ' '.join(current_content)
        })

    if not risks and paragraphs:
        for i, para in enumerate(paragraphs):
            if len(para) > 100:
                risks.append({
                    'header': f'Risk {i+1}',
                    'content': para
                })

    return risks


# Risk category keywords for preliminary classification
RISK_CATEGORIES = {
    'regulatory_legal': [
        r'regulat', r'compliance', r'legal', r'litigation', r'lawsuit',
        r'government', r'legislation', r'law ', r'court', r'patent',
        r'intellectual property', r'SEC', r'FDA', r'EPA', r'FTC'
    ],
    'cybersecurity': [
        r'cyber', r'data breach', r'security breach', r'hack', r'privacy',
        r'personal data', r'data protection', r'information security',
        r'GDPR', r'CCPA', r'ransomware', r'malware'
    ],
    'competitive_market': [
        r'compet', r'market share', r'pricing pressure', r'new entrants',
        r'industry consolidation', r'customer concentration', r'demand'
    ],
    'macroeconomic': [
        r'economic', r'recession', r'inflation', r'interest rate',
        r'currency', r'exchange rate', r'GDP', r'unemployment',
        r'global economy', r'trade war', r'tariff'
    ],
    'operational_supply': [
        r'supply chain', r'supplier', r'manufacturing', r'production',
        r'distribution', r'logistics', r'inventory', r'sourcing',
        r'operational', r'disruption'
    ],
    'financial_liquidity': [
        r'liquidity', r'cash flow', r'debt', r'credit', r'financing',
        r'capital', r'covenant', r'leverage', r'bankruptcy', r'insolven'
    ],
    'environmental_climate': [
        r'environment', r'climate', r'emission', r'carbon', r'pollution',
        r'sustainability', r'renewable', r'ESG', r'natural disaster',
        r'weather', r'flood', r'hurricane'
    ],
    'personnel_labor': [
        r'employee', r'personnel', r'labor', r'workforce', r'talent',
        r'key person', r'executive', r'management team', r'union',
        r'hiring', r'retention'
    ],
    'reputational': [
        r'reputation', r'brand', r'public perception', r'media',
        r'social media', r'negative publicity', r'trust'
    ],
    'technology_innovation': [
        r'technology', r'innovation', r'obsolete', r'R&D', r'research',
        r'product development', r'digital', r'AI ', r'artificial intelligence',
        r'automation', r'disrupt'
    ]
}


def classify_risk_paragraph(text):
    """Classify a risk paragraph into categories using keyword matching."""
    text_lower = text.lower()
    scores = {}
    for category, patterns in RISK_CATEGORIES.items():
        matches = sum(1 for p in patterns if re.search(p, text_lower))
        scores[category] = matches / len(patterns)
    return scores


def get_primary_category(scores):
    """Get the primary category from scores dict."""
    if not scores or max(scores.values()) == 0:
        return 'other'
    return max(scores, key=scores.get)
