"""
vehicle_matcher_brand_v1.py
----------------------------
Brand-aware ERP → DB vehicle model matching with improved generalization
"""

import re
from collections import defaultdict
from typing import List, Tuple, Optional, Set
import pandas as pd

# -----------------------------
# 1. FILE LOADING
# -----------------------------
def load_db_file(path: str, brand_col: str, model_col: str) -> List[Tuple[str, str]]:
    if path.endswith(".csv"):
        df = pd.read_csv(path, dtype=str)
    else:
        df = pd.read_excel(path, dtype=str)
    # strip whitespace and handle NaN
    df[brand_col] = df[brand_col].fillna('').astype(str).str.strip()
    df[model_col] = df[model_col].fillna('').astype(str).str.strip()
    # Filter out empty entries
    df = df[(df[brand_col] != '') & (df[model_col] != '')]
    return list(df[[brand_col, model_col]].itertuples(index=False, name=None))

def load_erp_file(path: str, brand_col: str, model_col: str) -> List[Tuple[str, str]]:
    if path.endswith(".csv"):
        df = pd.read_csv(path, dtype=str)
    else:
        df = pd.read_excel(path, dtype=str)
    # Handle NaN values and strip whitespace
    df[brand_col] = df[brand_col].fillna('').astype(str).str.strip()
    df[model_col] = df[model_col].fillna('').astype(str).str.strip()
    # Filter out entries where model is empty
    df = df[df[model_col] != '']
    return list(df[[brand_col, model_col]].itertuples(index=False, name=None))


# -----------------------------
# 2. NORMALIZE & TOKENIZE
# -----------------------------
def normalize_model_text(text: str) -> str:
    """
    Normalize model text to handle common variations
    Examples: '318 i' -> '318I', '730 li' -> '730LI'
    """
    if not text or text == 'nan':
        return ''
    
    text = text.upper().strip()
    
    # Common substitutions for better matching
    substitutions = {
        'SERIES': '',
        'TYPE': '',
        '-': '',
        '_': '',
        '/': ' ',
        '.': '',
    }
    
    for old, new in substitutions.items():
        text = text.replace(old, new)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize(text: str) -> List[str]:
    """Extract alphanumeric tokens, preserving order"""
    if not text or text == 'nan':
        return []
    normalized = normalize_model_text(text)
    return re.findall(r'[A-Z0-9]+', normalized)

def generate_variants(token: str) -> Set[str]:
    """
    Generate common variants of a token
    Examples: 
    - '318I' -> {'318I', '318', 'I318'}
    - '730LI' -> {'730LI', '730L', '730', 'LI730', 'L730'}
    - 'A4' -> {'A4', '4A'}
    """
    variants = {token}
    
    # Pattern: number + letters (e.g., 318I, 730LI, A4)
    match_num_letters = re.match(r'^(\d+)([A-Z]+)$', token)
    if match_num_letters:
        num, letters = match_num_letters.groups()
        variants.add(num)  # Just the number: 318
        
        # For each prefix of letters
        for i in range(1, len(letters) + 1):
            variants.add(num + letters[:i])  # 318I, 318IL, etc.
            variants.add(letters[:i] + num)  # I318, IL318, etc.
    
    # Pattern: letters + number (e.g., A4, S3)
    match_letters_num = re.match(r'^([A-Z]+)(\d+)$', token)
    if match_letters_num:
        letters, num = match_letters_num.groups()
        variants.add(num)  # Just the number
        variants.add(num + letters)  # 4A
        
        # For each prefix of letters
        for i in range(1, len(letters) + 1):
            variants.add(letters[:i] + num)  # A4, AB4, etc.
    
    # Pattern: mixed alphanumeric (e.g., C63AMG)
    # Extract number parts
    numbers = re.findall(r'\d+', token)
    letters_only = re.sub(r'\d+', '', token)
    
    if numbers and letters_only:
        for num in numbers:
            variants.add(num)
            variants.add(num + letters_only)
            variants.add(letters_only + num)
    
    return variants

_CODE_RE = re.compile(r'^([0-9]+[A-Z]{1,3}|[A-Z]{1,3}[0-9]+)$')
_ALWAYS_NOISE = {'LH', 'RH', 'STD', 'OPT', 'LHD', 'RHD', 'AUTO', 'MANUAL', 'MT', 'AT', 'CVT', 'DCT'}

def clean_erp_tokens(tokens: List[str], db_vocab: set, db_vocab_variants: set) -> List[str]:
    """Clean ERP tokens with improved logic and variant matching"""
    out = []
    seen = set()
    
    for t in tokens:
        t = t.upper()
        
        if t in seen:
            continue
        
        # Check if token or its variants exist in DB
        t_variants = generate_variants(t)
        has_variant_in_db = bool(t_variants & db_vocab_variants)
        
        if t in db_vocab or has_variant_in_db:
            out.append(t)
            seen.add(t)
        elif t in _ALWAYS_NOISE:
            continue
        elif t.isdigit():
            if len(t) >= 2:  # Keep meaningful numbers (e.g., 318, 730)
                out.append(t)
                seen.add(t)
        elif _CODE_RE.match(t) and not has_variant_in_db:
            # Only skip codes if they don't match DB variants
            continue
        elif len(t) <= 1:
            continue
        else:
            out.append(t)
            seen.add(t)
    
    return out


def build_db_vocab(db_entries: List[Tuple[str, str]]) -> tuple[set, set]:
    """
    Build vocabulary from all DB model tokens
    Returns: (vocab, vocab_with_variants)
    """
    vocab = set()
    vocab_with_variants = set()
    
    for _, model in db_entries:
        tokens = tokenize(model)
        vocab.update(tokens)
        
        # Also add all variants
        for token in tokens:
            vocab_with_variants.update(generate_variants(token))
    
    return vocab, vocab_with_variants

# -----------------------------
# 3. SCORING
# -----------------------------
def _fuzzy_match_score(token1: str, token2: str) -> float:
    """
    Calculate fuzzy match score between two tokens using variants
    Returns score between 0.0 and 1.0
    """
    if token1 == token2:
        return 1.0
    
    variants1 = generate_variants(token1)
    variants2 = generate_variants(token2)
    
    # Check for variant overlap
    overlap = variants1 & variants2
    if overlap:
        # Calculate quality of match based on variant similarity
        max_len = max(len(token1), len(token2))
        min_len = min(len(token1), len(token2))
        return 0.7 + (0.3 * min_len / max_len)  # 0.7 to 1.0 range
    
    # Check for substring matches
    if token1 in token2 or token2 in token1:
        if len(token1) >= 3 and len(token2) >= 3:
            return min(len(token1), len(token2)) / max(len(token1), len(token2)) * 0.6
    
    return 0.0

def _coverage(erp_tokens, db_tokens):
    """
    Calculate coverage considering fuzzy matches
    """
    if not db_tokens:
        return 0.0
    
    matched = 0.0
    for db_tok in db_tokens:
        best_match = 0.0
        for erp_tok in erp_tokens:
            score = _fuzzy_match_score(erp_tok, db_tok)
            best_match = max(best_match, score)
        matched += best_match
    
    return matched / len(db_tokens)

def _precision(erp_tokens, db_tokens):
    """
    Calculate precision considering fuzzy matches
    """
    if not erp_tokens:
        return 0.0
    
    matched = 0.0
    for erp_tok in erp_tokens:
        best_match = 0.0
        for db_tok in db_tokens:
            score = _fuzzy_match_score(erp_tok, db_tok)
            best_match = max(best_match, score)
        matched += best_match
    
    return matched / len(erp_tokens)

def _substring_bonus(erp_tokens, db_tokens):
    """Bonus for partial token matches"""
    best = 0.0
    for dt in db_tokens:
        for et in erp_tokens:
            if dt != et and len(dt) > 2 and len(et) > 2:
                if dt in et or et in dt:
                    overlap_ratio = min(len(dt), len(et)) / max(len(dt), len(et))
                    best = max(best, overlap_ratio * 0.10)
    return best

def _order_bonus(erp_tokens, db_tokens):
    """Bonus if tokens appear in same order"""
    if len(erp_tokens) < 2 or len(db_tokens) < 2:
        return 0.0
    
    # Find tokens that match (including fuzzy matches)
    erp_matched_indices = []
    db_matched_indices = []
    
    for i, erp_tok in enumerate(erp_tokens):
        for j, db_tok in enumerate(db_tokens):
            if _fuzzy_match_score(erp_tok, db_tok) >= 0.7:
                erp_matched_indices.append(i)
                db_matched_indices.append(j)
                break
    
    if len(erp_matched_indices) < 2:
        return 0.0
    
    # Check if relative order is preserved
    order_preserved = all(
        erp_matched_indices[i] < erp_matched_indices[i+1] and
        db_matched_indices[i] < db_matched_indices[i+1]
        for i in range(len(erp_matched_indices) - 1)
    )
    
    return 0.05 if order_preserved else 0.0

def score_model(erp_model_text: str, db_model_name: str, db_vocab: set, db_vocab_variants: set) -> float:
    """Calculate match score between ERP and DB model with fuzzy matching"""
    erp_tokens = clean_erp_tokens(tokenize(erp_model_text), db_vocab, db_vocab_variants)
    db_tokens = tokenize(db_model_name)
    
    if not db_tokens or not erp_tokens:
        return 0.0
    
    # Calculate fuzzy coverage and precision
    cov = _coverage(erp_tokens, db_tokens)
    prec = _precision(erp_tokens, db_tokens)
    
    # Additional bonuses
    sub = _substring_bonus(erp_tokens, db_tokens)
    order = _order_bonus(erp_tokens, db_tokens)
    
    # Bonus for similar token counts
    len_diff = abs(len(erp_tokens) - len(db_tokens))
    len_bonus = 0.05 if len_diff == 0 else (0.03 if len_diff == 1 else 0.0)
    
    # Penalty for too many unmatched tokens
    extra_tokens = max(len(erp_tokens) - len(db_tokens), 0)
    extra_penalty = min(extra_tokens * 0.02, 0.10)
    
    # Weighted combination
    score = (cov * 0.55 + prec * 0.30 + sub + order + len_bonus - extra_penalty) * 100
    return max(score, 0.0)

# -----------------------------
# 4. INDEX
# -----------------------------
class ModelIndex:
    def __init__(self, db_entries: List[Tuple[str, str]]):
        self.db_entries = db_entries
        self.db_vocab, self.db_vocab_variants = build_db_vocab(db_entries)
        self._token_idx = defaultdict(list)
        self._variant_idx = defaultdict(list)
        self._brand_idx = defaultdict(list)
        
        for i, (brand, model) in enumerate(db_entries):
            tokens = tokenize(model)
            
            # Index by exact tokens
            for tok in tokens:
                self._token_idx[tok].append(i)
                
                # Also index by all variants
                for variant in generate_variants(tok):
                    self._variant_idx[variant].append(i)
            
            self._brand_idx[brand.upper()].append(i)

    def candidates(self, erp_model_text: str, erp_brand: Optional[str]) -> List[Tuple[str, str]]:
        """Get candidate DB entries based on token and variant overlap"""
        erp_toks = tokenize(erp_model_text)
        
        if not erp_toks:
            return []
        
        seen = set()
        
        # Match by exact tokens and variants
        for t in erp_toks:
            # Exact match
            for idx in self._token_idx.get(t, []):
                seen.add(idx)
            
            # Variant match
            for variant in generate_variants(t):
                for idx in self._variant_idx.get(variant, []):
                    seen.add(idx)
        
        # If brand provided, filter by brand
        if erp_brand and erp_brand.strip():
            brand_indices = set(self._brand_idx.get(erp_brand.upper(), []))
            if brand_indices:
                seen &= brand_indices
        
        return [self.db_entries[i] for i in seen]

# -----------------------------
# 5. MATCHER
# -----------------------------
def match_vehicle_models(
    erp_brand: str,
    erp_model_text: str,
    db_entries: List[Tuple[str, str]],
    threshold: float = 40,
    index: Optional[ModelIndex] = None,
) -> Tuple[Optional[Tuple[str, str]], float, bool]:
    """
    Match ERP model to DB model with fuzzy matching
    Returns: (best_match, score, is_brand_mismatch)
    """
    db_vocab = index.db_vocab if index else build_db_vocab(db_entries)[0]
    db_vocab_variants = index.db_vocab_variants if index else build_db_vocab(db_entries)[1]
    
    # Get candidates - try brand-specific first, then all
    pool = []
    if index:
        pool = index.candidates(erp_model_text, erp_brand)
        # If no brand-specific matches, try without brand filter
        if not pool and erp_brand:
            pool = index.candidates(erp_model_text, None)
    else:
        pool = db_entries
    
    if not pool:
        return None, 0.0, False
    
    erp_tokens = clean_erp_tokens(tokenize(erp_model_text), db_vocab, db_vocab_variants)
    candidates = []

    for db_brand, db_model in pool:
        s = score_model(erp_model_text, db_model, db_vocab, db_vocab_variants)
        if s > 0:
            is_mismatch = (erp_brand and erp_brand.upper() != db_brand.upper())
            candidates.append(((db_brand, db_model), s, is_mismatch))

    if not candidates:
        return None, 0.0, False

    def sort_key(item):
        (db_brand, db_model), s, is_mismatch = item
        db_toks = tokenize(db_model)
        
        # Count fuzzy matches
        matched_count = sum(
            1 for db_tok in db_toks
            if any(_fuzzy_match_score(erp_tok, db_tok) >= 0.7 for erp_tok in erp_tokens)
        )
        
        # Prioritize: brand match > score > matched tokens > fewer total tokens
        brand_match_priority = 0 if is_mismatch else 1
        return (brand_match_priority, round(s, 1), matched_count, -len(db_toks))

    candidates.sort(key=sort_key, reverse=True)
    best_entry, best_score, is_mismatch = candidates[0]
    
    return (best_entry, round(best_score, 1), is_mismatch) if best_score >= threshold else (None, 0.0, False)

# -----------------------------
# 6. RUN MATCHING + SAVE
# -----------------------------
def run_matching(db_path, erp_path, db_brand_col, db_model_col, erp_brand_col, erp_model_col, threshold=40, output_file="3_MONTH_D.xlsx"):
    print("Loading database...")
    db_entries = load_db_file(db_path, db_brand_col, db_model_col)
    print(f"Loaded {len(db_entries)} database entries")
    
    print("Loading ERP data...")
    erp_inputs = load_erp_file(erp_path, erp_brand_col, erp_model_col)
    print(f"Loaded {len(erp_inputs)} ERP entries")
    
    print("Building index with variants...")
    idx = ModelIndex(db_entries)
    print(f"Index built with {len(idx.db_vocab)} unique tokens and {len(idx.db_vocab_variants)} variants")

    results = []
    brand_mismatch = []
    zero_scored = []
    high_confidence = []

    print("Matching...")
    for i, (erp_brand, erp_model) in enumerate(erp_inputs):
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(erp_inputs)} entries...")
        
        # Combine brand and model for matching
        combined_text = f"{erp_brand} {erp_model}" if erp_brand else erp_model
        match, score, is_brand_mismatch = match_vehicle_models(erp_brand, combined_text, db_entries, threshold, idx)
        
        if score == 0:
            zero_scored.append({
                "ERP_Brand": erp_brand,
                "ERP_Model": erp_model
            })

        if match:
            db_brand, db_model = match
            
            # Apply brand mismatch penalty
            if is_brand_mismatch:
                score = max(score - 30, 0)
                brand_mismatch.append({
                    "ERP_Brand": erp_brand,
                    "ERP_Model": erp_model,
                    "Matched_Brand": db_brand,
                    "Matched_Model": db_model,
                    "Score": score
                })
            
            # Add to high confidence list ONLY if score >= 60 AND no brand mismatch
            if score >= 60 and not is_brand_mismatch:
                high_confidence.append({
                    "ERP_Brand": erp_brand,
                    "ERP_Model": erp_model,
                    "Matched_Brand": db_brand,
                    "Matched_Model": db_model,
                    "Score": score
                })
        else:
            db_brand, db_model = None, None

        results.append({
            "ERP_Brand": erp_brand,
            "ERP_Model": erp_model,
            "Matched_Brand": db_brand,
            "Matched_Model": db_model,
            "Score": score
        })

    print("\nSaving results...")
    # Save all sheets in one Excel file
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        pd.DataFrame(results).to_excel(writer, sheet_name="All_Matches", index=False)
        pd.DataFrame(high_confidence).to_excel(writer, sheet_name="High_Confidence_60+", index=False)
        pd.DataFrame(brand_mismatch).to_excel(writer, sheet_name="Brand_Mismatches", index=False)
        pd.DataFrame(zero_scored).to_excel(writer, sheet_name="Zero_Score_Matches", index=False)

    print(f"\nSaved mapping results to {output_file}")
    print(f"Total matches: {sum(1 for r in results if r['Score'] > 0)}/{len(results)}")
    print(f"High confidence matches (score >= 60, no brand mismatch): {len(high_confidence)}")
    print(f"Brand mismatches: {len(brand_mismatch)}")
    print(f"Zero score matches: {len(zero_scored)}")
    
    return pd.DataFrame(results), pd.DataFrame(brand_mismatch), pd.DataFrame(high_confidence)


# -----------------------------
# 7. MAIN
# -----------------------------
if __name__ == "__main__":
    db_file_path = "after_filling_final.csv"
    erp_file_path = "ERP_D.xlsx"

    db_brand_col = "BRAND"
    db_model_col = "MODEL"
    erp_brand_col = "m_name"
    erp_model_col = "MODEL_NAME"

    run_matching(db_file_path, erp_file_path, db_brand_col, db_model_col, erp_brand_col, erp_model_col)