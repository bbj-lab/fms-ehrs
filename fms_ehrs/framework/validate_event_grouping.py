#!/usr/bin/env python3
"""
Validation Script for Event Token Grouping

Validates that quantile and text tokens appear in the correct positions.

Usage:
1. Paste your timeline into the TIMELINE variable below (with timestamps appended to tokens)
2. Run: python validate_event_grouping.py
"""

import re
from typing import List, Optional
from dataclasses import dataclass

# =============================================================================
# CONFIGURATION - UPDATE THIS TIMELINE
# =============================================================================

# Paste your timeline here (with timestamps appended to tokens)
TIMELINE = ['TL_START_2147-12-10T22:49:00', 'RACE_unknown_2147-12-10T22:49:00', 'SEX_f_2147-12-10T22:49:00', 'Q4_2147-12-10T22:49:00', 'ADMN_ew_emer._2147-12-10T22:49:00', 'LAB_50931_2147-12-11T06:10:00', 'Q9_2147-12-11T06:10:00', 'LAB_50970_2147-12-11T06:10:00', 'Q6_2147-12-11T06:10:00', 'LAB_50911_2147-12-11T06:10:00', 'Q4_2147-12-11T06:10:00', 'LAB_51010_2147-12-11T06:10:00', 'Q2_2147-12-11T06:10:00', 'LAB_51249_2147-12-11T06:10:00', 'Q9_2147-12-11T06:10:00', 'LAB_50983_2147-12-11T06:10:00', 'Q5_2147-12-11T06:10:00', 'LAB_50882_2147-12-11T06:10:00', 'Q6_2147-12-11T06:10:00', 'LAB_51237_2147-12-11T06:10:00', 'Q0_2147-12-11T06:10:00', 'LAB_50893_2147-12-11T06:10:00', 'Q8_2147-12-11T06:10:00', 'LAB_50905_2147-12-11T06:10:00', 'Q9_2147-12-11T06:10:00', 'LAB_51250_2147-12-11T06:10:00', 'Q2_2147-12-11T06:10:00', 'LAB_51613_2147-12-11T06:10:00', 'Q9_2147-12-11T06:10:00', 'LAB_51277_2147-12-11T06:10:00', 'Q0_2147-12-11T06:10:00', 'LAB_50904_2147-12-11T06:10:00', 'Q5_2147-12-11T06:10:00', 'LAB_51221_2147-12-11T06:10:00', 'Q9_2147-12-11T06:10:00', 'LAB_50912_2147-12-11T06:10:00', 'Q3_2147-12-11T06:10:00', 'LAB_51248_2147-12-11T06:10:00', 'Q5_2147-12-11T06:10:00', 'LAB_50878_2147-12-11T06:10:00', 'Q1_2147-12-11T06:10:00', 'LAB_51279_2147-12-11T06:10:00', 'Q9_2147-12-11T06:10:00', 'LAB_51006_2147-12-11T06:10:00', 'Q4_2147-12-11T06:10:00', 'LAB_51222_2147-12-11T06:10:00', 'Q9_2147-12-11T06:10:00', 'LAB_51003_2147-12-11T06:10:00', 'LAB_51274_2147-12-11T06:10:00', 'Q0_2147-12-11T06:10:00', 'LAB_50852_2147-12-11T06:10:00', 'Q9_2147-12-11T06:10:00', 'LAB_50863_2147-12-11T06:10:00', 'Q2_2147-12-11T06:10:00', 'LAB_51301_2147-12-11T06:10:00', 'Q5_2147-12-11T06:10:00', 'LAB_50903_2147-12-11T06:10:00', 'Q9_2147-12-11T06:10:00', 'LAB_50902_2147-12-11T06:10:00', 'Q4_2147-12-11T06:10:00', 'LAB_51000_2147-12-11T06:10:00', 'Q9_2147-12-11T06:10:00', 'LAB_50861_2147-12-11T06:10:00', 'Q3_2147-12-11T06:10:00', 'LAB_51275_2147-12-11T06:10:00', 'Q2_2147-12-11T06:10:00', 'LAB_50960_2147-12-11T06:10:00', 'Q9_2147-12-11T06:10:00', 'LAB_50868_2147-12-11T06:10:00', 'Q7_2147-12-11T06:10:00', 'LAB_50885_2147-12-11T06:10:00', 'Q6_2147-12-11T06:10:00', 'LAB_50971_2147-12-11T06:10:00', 'Q7_2147-12-11T06:10:00', 'LAB_51265_2147-12-11T06:10:00', 'Q7_2147-12-11T06:10:00', 'LAB_50910_2147-12-11T06:10:00', 'Q5_2147-12-11T06:10:00', 'LAB_50907_2147-12-11T06:10:00', 'Q9_2147-12-11T06:10:00', 'LAB_50906_2147-12-11T06:10:00', 'Q7_2147-12-11T06:10:00', 'PROC_3812_2147-12-12T00:00:00', 'PROC_0040_2147-12-12T00:00:00', 'LAB_51274_2147-12-12T15:20:00', 'Q0_2147-12-12T15:20:00', 'LAB_51237_2147-12-12T15:20:00', 'Q1_2147-12-12T15:20:00', 'LAB_51275_2147-12-12T15:20:00', 'Q9_2147-12-12T15:20:00', 'LAB_50960_2147-12-13T02:41:00', 'Q0_2147-12-13T02:41:00', 'LAB_51279_2147-12-13T02:41:00', 'Q7_2147-12-13T02:41:00', 'LAB_51248_2147-12-13T02:41:00', 'Q4_2147-12-13T02:41:00', 'LAB_51277_2147-12-13T02:41:00', 'Q0_2147-12-13T02:41:00', 'LAB_51221_2147-12-13T02:41:00', 'Q7_2147-12-13T02:41:00', 'LAB_51222_2147-12-13T02:41:00', 'Q7_2147-12-13T02:41:00', 'LAB_51250_2147-12-13T02:41:00', 'Q2_2147-12-13T02:41:00', 'LAB_51249_2147-12-13T02:41:00', 'Q9_2147-12-13T02:41:00', 'LAB_51301_2147-12-13T02:41:00', 'Q8_2147-12-13T02:41:00', 'LAB_50868_2147-12-13T02:41:00', 'Q7_2147-12-13T02:41:00', 'LAB_50931_2147-12-13T02:41:00', 'Q9_2147-12-13T02:41:00', 'LAB_50983_2147-12-13T02:41:00', 'Q2_2147-12-13T02:41:00', 'LAB_50902_2147-12-13T02:41:00', 'Q4_2147-12-13T02:41:00', 'LAB_50971_2147-12-13T02:41:00', 'Q4_2147-12-13T02:41:00', 'LAB_50882_2147-12-13T02:41:00', 'Q2_2147-12-13T02:41:00', 'LAB_50970_2147-12-13T02:41:00', 'Q8_2147-12-13T02:41:00', 'LAB_51265_2147-12-13T02:41:00', 'Q6_2147-12-13T02:41:00', 'LAB_51006_2147-12-13T02:41:00', 'Q4_2147-12-13T02:41:00', 'LAB_50983_2147-12-14T05:15:00', 'Q5_2147-12-14T05:15:00', 'LAB_51222_2147-12-14T05:15:00', 'Q8_2147-12-14T05:15:00', 'LAB_51221_2147-12-14T05:15:00', 'Q8_2147-12-14T05:15:00', 'LAB_50912_2147-12-14T05:15:00', 'Q2_2147-12-14T05:15:00', 'LAB_50893_2147-12-14T05:15:00', 'Q5_2147-12-14T05:15:00', 'LAB_50931_2147-12-14T05:15:00', 'Q8_2147-12-14T05:15:00', 'LAB_51006_2147-12-14T05:15:00', 'Q1_2147-12-14T05:15:00', 'LAB_50960_2147-12-14T05:15:00', 'Q6_2147-12-14T05:15:00', 'LAB_51249_2147-12-14T05:15:00', 'Q8_2147-12-14T05:15:00', 'LAB_51250_2147-12-14T05:15:00', 'Q2_2147-12-14T05:15:00', 'LAB_51279_2147-12-14T05:15:00', 'Q8_2147-12-14T05:15:00', 'LAB_50882_2147-12-14T05:15:00', 'Q5_2147-12-14T05:15:00', 'LAB_51277_2147-12-14T05:15:00', 'Q0_2147-12-14T05:15:00', 'LAB_50970_2147-12-14T05:15:00', 'Q7_2147-12-14T05:15:00', 'LAB_50971_2147-12-14T05:15:00', 'Q3_2147-12-14T05:15:00', 'LAB_51301_2147-12-14T05:15:00', 'Q6_2147-12-14T05:15:00', 'LAB_50902_2147-12-14T05:15:00', 'Q6_2147-12-14T05:15:00', 'LAB_50868_2147-12-14T05:15:00', 'Q5_2147-12-14T05:15:00', 'LAB_51248_2147-12-14T05:15:00', 'Q4_2147-12-14T05:15:00', 'LAB_51265_2147-12-14T05:15:00', 'Q7_2147-12-14T05:15:00', 'DSCG_rehab_2147-12-17T13:30:00', 'TL_END_2147-12-17T13:30:00']
# =============================================================================
# PARSING
# =============================================================================

@dataclass
class Token:
    """Represents a token in a timeline"""
    position: int
    token: str  # Full token with timestamp
    base: str  # Token without timestamp (e.g., "LAB_50934")
    token_type: str  # 'code', 'quantile', 'text', or 'other'
    
    def __repr__(self):
        return f"{self.base}"

def parse_token(position: int, full_token: str) -> Token:
    """Parse a token into its components"""
    # Split token and timestamp (format: TOKEN_TIMESTAMP)
    parts = full_token.rsplit('_', 1)
    if len(parts) == 2 and len(parts[1]) > 10:  # Timestamp is long
        base_token = parts[0]
    else:
        base_token = full_token
    
    # Determine token type
    token_type = "other"
    
    # Match quantile tokens: Q0-Q9
    if re.match(r'^Q[0-9]$', base_token):
        token_type = "quantile"
    # Match text tokens: TEXT_*
    elif base_token.startswith('TEXT_'):
        token_type = "text"
    # Match code tokens: PREFIX_CODE (e.g., LAB_50934, VTL_123)
    elif re.match(r'^[A-Z]+_[A-Z0-9]+', base_token):
        token_type = "code"
    
    return Token(
        position=position,
        token=full_token,
        base=base_token,
        token_type=token_type
    )

# =============================================================================
# VALIDATION
# =============================================================================

def validate_quantile_positions(tokens: List[Token]) -> List[dict]:
    """
    Check error case 1: Quantile tokens must appear immediately after a code token.
    
    Valid patterns:
    - CODE, QUANTILE (quantile belongs to code)
    - CODE, CODE, QUANTILE (quantile belongs to second code, first code has no quantile)
    - CODE, TEXT (valid, code with text but no quantile)
    - CODE, QUANTILE, TEXT (valid sequence)
    
    Invalid patterns:
    - QUANTILE at start (orphaned quantile)
    - TEXT, QUANTILE (text should come after quantile)
    - CODE, TEXT, QUANTILE (wrong order)
    - QUANTILE, QUANTILE (consecutive quantiles, unless preceded by code)
    
    Returns list of errors found.
    """
    errors = []
    
    # Check 3-token patterns
    for i in range(len(tokens) - 2):
        t1 = tokens[i]
        t2 = tokens[i + 1]
        t3 = tokens[i + 2]
        
        # Error: CODE, TEXT, QUANTILE - text shouldn't come before quantile
        if t1.token_type == "code" and t2.token_type == "text" and t3.token_type == "quantile":
            errors.append({
                'position': i,
                'error': 'WRONG_TOKEN_ORDER',
                'pattern': 'CODE-TEXT-QUANTILE',
                'tokens': [t1, t2, t3]
            })
        
        # Error: NON_CODE, QUANTILE, QUANTILE - two consecutive quantiles not after code
        if t1.token_type != "code" and t2.token_type == "quantile" and t3.token_type == "quantile":
            errors.append({
                'position': i,
                'error': 'CONSECUTIVE_QUANTILES',
                'pattern': 'NON_CODE-QUANTILE-QUANTILE',
                'tokens': [t1, t2, t3]
            })
    
    # Check 2-token patterns
    for i in range(len(tokens) - 1):
        t1 = tokens[i]
        t2 = tokens[i + 1]
        
        # Error: TEXT, QUANTILE - text should come after quantile
        if t1.token_type == "text" and t2.token_type == "quantile":
            errors.append({
                'position': i,
                'error': 'WRONG_TOKEN_ORDER',
                'pattern': 'TEXT-QUANTILE',
                'tokens': [t1, t2]
            })
    
    # Check for quantiles at start of timeline (orphaned quantiles)
    if tokens and tokens[0].token_type == "quantile":
        errors.append({
            'position': 0,
            'error': 'ORPHANED_QUANTILE',
            'pattern': 'QUANTILE_AT_START',
            'tokens': [tokens[0]]
        })
    
    # Check for consecutive quantiles not preceded by code
    for i in range(len(tokens) - 1):
        if tokens[i].token_type == "quantile" and tokens[i + 1].token_type == "quantile":
            # Check if previous token was a code
            if i == 0 or tokens[i - 1].token_type != "code":
                errors.append({
                    'position': i,
                    'error': 'CONSECUTIVE_QUANTILES',
                    'pattern': 'QUANTILE-QUANTILE',
                    'tokens': [tokens[i], tokens[i + 1]]
                })
    
    return errors

def validate_text_positions(tokens: List[Token]) -> List[dict]:
    """
    Check error case 2: Text tokens must appear immediately after a code or quantile token.
    
    Error pattern: CODE, CODE, TEXT or QUANTILE, QUANTILE, TEXT (ambiguous ownership)
    
    Returns list of errors found.
    """
    errors = []
    
    for i in range(len(tokens) - 2):
        t1 = tokens[i]
        t2 = tokens[i + 1]
        t3 = tokens[i + 2]
        
        # Error case: Two code tokens followed by a text token
        if t1.token_type == "code" and t2.token_type == "code" and t3.token_type == "text":
            errors.append({
                'position': i,
                'error': 'TEXT_NOT_AFTER_CODE_OR_QUANTILE',
                'pattern': 'CODE-CODE-TEXT',
                'tokens': [t1, t2, t3]
            })
        
        # Error case: Two quantile tokens followed by a text token
        if t1.token_type == "quantile" and t2.token_type == "quantile" and t3.token_type == "text":
            errors.append({
                'position': i,
                'error': 'TEXT_NOT_AFTER_CODE_OR_QUANTILE',
                'pattern': 'QUANTILE-QUANTILE-TEXT',
                'tokens': [t1, t2, t3]
            })
    
    # Check for text at start of timeline (orphaned text)
    if tokens and tokens[0].token_type == "text":
        errors.append({
            'position': 0,
            'error': 'ORPHANED_TEXT',
            'pattern': 'TEXT_AT_START',
            'tokens': [tokens[0]]
        })
    
    # Check for consecutive text tokens (orphaned text)
    for i in range(len(tokens) - 1):
        if tokens[i].token_type == "text" and tokens[i + 1].token_type == "text":
            # Check if previous token was a code or quantile
            if i == 0 or tokens[i - 1].token_type not in ["code", "quantile"]:
                errors.append({
                    'position': i,
                    'error': 'CONSECUTIVE_TEXT',
                    'pattern': 'TEXT-TEXT',
                    'tokens': [tokens[i], tokens[i + 1]]
                })
    
    return errors

def validate_timeline(timeline: List[str]):
    """Main validation function"""
    if not timeline:
        print("❌ Error: TIMELINE is empty. Please paste your timeline into the TIMELINE variable.")
        return
    
    print("=" * 80)
    print("🔍 EVENT TOKEN GROUPING VALIDATION")
    print("=" * 80)
    
    # Parse all tokens
    tokens = [parse_token(i, token) for i, token in enumerate(timeline)]
    
    print(f"\n📊 Total tokens: {len(tokens)}")
    
    # Count token types
    from collections import Counter
    type_counts = Counter(t.token_type for t in tokens)
    print("\nToken type distribution:")
    for token_type, count in sorted(type_counts.items()):
        print(f"  {token_type}: {count}")
    
    # Validate
    quantile_errors = validate_quantile_positions(tokens)
    text_errors = validate_text_positions(tokens)
    
    all_errors = quantile_errors + text_errors
    
    print("\n" + "=" * 80)
    if not all_errors:
        print("✅ VALIDATION PASSED: No errors found!")
    else:
        print(f"❌ VALIDATION FAILED: Found {len(all_errors)} error(s)")
        print("\n🚨 ERRORS DETECTED:")
        print("=" * 80)
        
        for idx, error in enumerate(all_errors, 1):
            print(f"\nError {idx}: {error['error']}")
            print(f"  Pattern: {error['pattern']}")
            print(f"  Position: {error['position']}")
            print(f"  Tokens:")
            for token in error['tokens']:
                print(f"    Position {token.position}: {token.base} ({token.token_type})")
            
            # Show context (5 tokens before and after)
            pos = error['position']
            start = max(0, pos - 5)
            end = min(len(tokens), pos + len(error['tokens']) + 5)
            print(f"\n  Context (positions {start}-{end-1}):")
            for i in range(start, end):
                marker = ">>>" if i == pos else "   "
                print(f"    {marker} {i:3d}: {tokens[i].base}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print(f"  Total tokens analyzed: {len(tokens)}")
    print(f"  Quantile errors: {len(quantile_errors)}")
    print(f"  Text errors: {len(text_errors)}")
    print(f"  Total errors: {len(all_errors)}")
    print("=" * 80)

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the validation"""
    validate_timeline(TIMELINE)

if __name__ == "__main__":
    main()

