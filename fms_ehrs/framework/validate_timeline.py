#!/usr/bin/env python3
"""
Simple Timeline Validation Script for MIMIC-IV Data

Usage:
1. Set the HADM_ID and TIMELINE variables below
2. Update the MIMIC_DATA_PATH to point to your MIMIC-IV data
3. Run: python validate_timeline_simple.py
"""

import polars as pl
from pathlib import Path
import re
from typing import List, Dict, Tuple, Optional

# =============================================================================
# CONFIGURATION - UPDATE THESE VALUES
# =============================================================================

# Path to your MIMIC-IV data directory
# MIMIC_DATA_PATH = "/home/chend5/fms-ehrs/data/mimic-iv-2.2"
MIMIC_DATA_PATH = "/gpfs/data/bbj-lab/data/physionet.org/files/mimiciv_parquet"

# Hospital Admission ID to validate against
HADM_ID = 28289260

# Timeline to validate (replace with your actual timeline)
TIMELINE =  ['TL_START', 'RACE_white', 'SEX_f', 'Q1', 'ADMN_ew_emer.', 'LAB_51254', 'Q5', 'LAB_51466', 'TEXT_NEG', 
             'LAB_50878', 'Q6', 'LAB_50931', 'Q4', 'LAB_50912', 'Q1', 'LAB_51265', 'Q2', 'LAB_51237', 'Q8', 'LAB_50843', 
             'Q5', 'LAB_51097', 'Q9', 'LAB_51249', 'Q4', 'LAB_51100', 'LAB_51006', 'Q1', 'LAB_50983', 'Q0', 
             'LAB_51200', 'Q5', 'LAB_51120', 'Q5', 'LAB_51487', 'TEXT_NEG', 'LAB_51250', 'Q9', 'LAB_50956', 'Q9', 
             'LAB_51108', 'Q9', 'LAB_50931', 'Q5', 'LAB_51222', 'Q6', 'LAB_51274', 'Q8', 'LAB_52065', 'Q1', 'LAB_51248', 'Q9', 
             'LAB_51277', 'Q3', 'LAB_50902', 'Q1', 'LAB_51301', 'Q9', 'LAB_51078', 'LAB_51117', 'Q9', 'LAB_50885', 'Q9', 
             'LAB_51508', 'TEXT_Amber', 'LAB_51085', 'TEXT_NEG', 'LAB_51116', 'Q1', 'LAB_51476', 'TEXT_1120', 'LAB_51265', 'Q1', 
             'LAB_51120', 'Q7', 'LAB_51118', 'Q8', 'LAB_51250', 'Q9', 'LAB_50861', 'Q4', 'LAB_50955', 'LAB_51514', 'Q1', 
             'LAB_52065', 'Q5', 'LAB_51221', 'Q3', 'LAB_50878', 'Q8', 'LAB_50868', 'Q5', 'LAB_51249', 'Q5', 'LAB_51221', 'Q6', 
             'LAB_51249', 'Q4', 'LAB_51006', 'Q0', 'LAB_50960', 'Q5', 'LAB_51265', 'Q3', 'LAB_51118', 'Q9', 'LAB_50902', 'Q2', 
             'LAB_50971', 'Q4', 'LAB_50878', 'Q6', 'LAB_51277', 'Q3', 'LAB_51491', 'Q8', 'LAB_50861', 'Q4', 'LAB_51125', 'Q7', 
             'LAB_51125', 'Q5', 'LAB_51127', 'Q8', 'LAB_50835', 'LAB_50933', 'LAB_51519', 'TEXT_NONE', 'LAB_51256', 'Q5', 
             'LAB_50861', 'Q4', 'LAB_51493', 'TEXT_02', 'LAB_51093', 'Q6', 'LAB_51274', 'Q8', 'LAB_51067', 'Q9', 
             'LAB_50893', 'Q0', 'LAB_51222', 'Q4', 'LAB_51127', 'Q2', 'LAB_51250', 'Q9', 'LAB_51082', 'Q8', 'LAB_51275', 'Q5', 
             'LAB_50920', 'LAB_51463', 'TEXT_FEW', 'LAB_51082', 'Q9', 'LAB_50983', 'Q0', 'LAB_50868', 'Q1', 'LAB_50863', 'Q6', 
             'LAB_51248', 'Q9', 'LAB_51087', 'Q9', 'LAB_51117', 'Q7', 'LAB_51221', 'Q4', 'LAB_50868', 'Q5', 'LAB_50863', 'Q4', 
             'LAB_51277', 'Q3', 'LAB_50983', 'Q0', 'LAB_51265', 'Q1', 'LAB_51114', 'Q9', 'LAB_50971', 'Q7', 'LAB_50882', 'Q5', 
             'LAB_50954', 'Q3', 'LAB_51277', 'Q3', 'LAB_50863', 'Q5', 'LAB_51250', 'Q9', 'LAB_51274', 'Q9', 'LAB_50885', 'Q8', 
             'LAB_50954', 'Q7', 'LAB_51221', 'Q4', 'LAB_51498', 'Q6', 'LAB_51279', 'Q2', 'LAB_51146', 'Q5', 'LAB_51279', 'Q1', 
             'LAB_51006', 'Q1', 'LAB_50882', 'Q5', 'LAB_51237', 'Q8', 'LAB_51486', 'TEXT_TR', 'LAB_50849', 'Q2', 'LAB_51301', 
             'Q6', 'LAB_51249', 'Q5', 'LAB_50902', 'Q0', 'LAB_51222', 'Q4', 'LAB_51006', 'Q0', 'LAB_51244', 'Q4', 'LAB_51275', 
             'Q6', 'LAB_51279', 'Q3', 'LAB_51248', 'Q9', 'LAB_50970', 'Q7', 'LAB_50862', 'Q3', 'LAB_51492', 'Q1', 
             'LAB_51237', 'Q9', 'LAB_51222', 'Q4', 'LAB_51301', 'Q8', 'LAB_51506', 'TEXT_Clear', 'LAB_51279', 'Q1', 
             'LAB_50885', 'Q6', 'LAB_51248', 'Q9', 'LAB_50862', 'Q1', 'LAB_51482', 'TEXT_02', 'LAB_51301', 'Q8', 
             'LAB_51484', 'TEXT_TR', 'LAB_50882', 'Q6', 'LAB_51464', 'TEXT_SM', 'LAB_50912', 'Q1', 'LAB_51116', 'Q0', 
             'LAB_51516', 'TEXT_610', 'LAB_50931', 'Q7', 'LAB_51087', 'LAB_50912', 'Q2', 'LAB_51221', 'Q5', 'LAB_50971', 'Q3', 
             'LAB_50912', 'Q1', 'LAB_51104', 'Q6', 'LAB_51094', 'Q9', 'LAB_51478', 'TEXT_NEG', 'LAB_50862', 'Q0', 'LAB_51512', 
             'TEXT_0CC', 'PROC_5491', 'DSCG_home', 'TL_END']
# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def extract_lab_tokens(timeline: List[str]) -> List[Tuple[int, int, Optional[int]]]:
    """
    Extract lab tokens and their quantiles from timeline
    
    Returns:
        List of tuples: (position, itemid, quantile)
    """
    lab_tokens = []
    i = 0
    
    while i < len(timeline):
        token = timeline[i]
        
        if token.startswith('LAB_'):
            try:
                itemid = int(token.split('_')[1])
                
                # Check if next token is a quantile
                quantile = None
                if i + 1 < len(timeline):
                    next_token = timeline[i + 1]
                    if next_token.startswith('Q') and len(next_token) == 2:
                        quantile = int(next_token[1])
                        i += 1  # Skip the quantile token
                
                lab_tokens.append((i, itemid, quantile))
            except (ValueError, IndexError):
                print(f"Warning: Could not parse lab token: {token}")
        
        i += 1
    
    return lab_tokens

def get_lab_data(hadm_id: int, itemids: List[int], mimic_path: str) -> pl.DataFrame:
    """Get actual lab data from MIMIC-IV"""
    labevents_path = Path(mimic_path) / "hosp" / "labevents.parquet"
    
    if not labevents_path.exists():
        raise FileNotFoundError(f"Lab events file not found: {labevents_path}")
    
    lab_data = (
        pl.scan_parquet(labevents_path)
        .filter(
            (pl.col("hadm_id") == hadm_id) & 
            (pl.col("itemid").is_in(itemids))
        )
        .select([
            "itemid",
            "charttime", 
            "value",
            "valueuom",
            "ref_range_lower",
            "ref_range_upper"
        ])
        .sort(["itemid", "charttime"])
        .collect()
    )
    
    return lab_data

def get_lab_item_names(itemids: List[int], mimic_path: str) -> Dict[int, str]:
    """Get lab item names from d_labitems"""
    d_labitems_path = Path(mimic_path) / "hosp" / "d_labitems.parquet"
    
    if not d_labitems_path.exists():
        return {}
    
    try:
        lab_items = (
            pl.scan_parquet(d_labitems_path)
            .filter(pl.col("itemid").is_in(itemids))
            .select(["itemid", "label"])
            .collect()
        )
        
        return {row["itemid"]: row["label"] for row in lab_items.to_dicts()}
    except Exception as e:
        print(f"Warning: Could not load lab items: {e}")
        return {}

def validate_timeline(timeline: List[str], hadm_id: int, mimic_path: str):
    """Main validation function"""
    
    print(f"🔍 Validating timeline for hadm_id: {hadm_id}")
    print(f"📊 Timeline length: {len(timeline)} tokens")
    
    # Extract lab tokens
    lab_tokens = extract_lab_tokens(timeline)
    print(f"🧪 Found {len(lab_tokens)} lab tokens")
    
    if not lab_tokens:
        print("❌ No lab tokens found in timeline")
        return
    
    # Get unique itemids
    itemids = list(set([itemid for _, itemid, _ in lab_tokens]))
    print(f"🔬 Unique lab itemids: {len(itemids)}")
    
    # Get actual lab data
    try:
        actual_data = get_lab_data(hadm_id, itemids, mimic_path)
        print(f"📈 Found {len(actual_data)} actual lab results")
    except Exception as e:
        print(f"❌ Error getting lab data: {e}")
        return
    
    # Get lab item names
    lab_names = get_lab_item_names(itemids, mimic_path)
    
    # Order validation helper: ensure LAB token order follows charttime order
    def validate_lab_ordering(lab_tokens: List[Tuple[int, int, Optional[int]]], actual_data: pl.DataFrame) -> None:
        print("\n⏱️ ORDER VALIDATION (LAB tokens vs labevents.charttime):")
        print("=" * 80)
        if actual_data.is_empty():
            print("No actual lab data available to validate ordering.")
            return
        # Build mapping itemid -> sorted list of charttimes
        itemid_to_times: Dict[int, List] = {}
        for row in (
            actual_data
            .select(["itemid", "charttime"])  # ensure only required cols
            .sort(["itemid", "charttime"])   # global sort
        ).to_dicts():
            itemid_to_times.setdefault(row["itemid"], []).append(row["charttime"])        
        matched = 0
        missing = 0
        out_of_order = 0
        last_time = None
        examples: List[str] = []
        for pos, itemid, _ in lab_tokens:
            times = itemid_to_times.get(itemid, [])
            if not times:
                missing += 1
                examples.append(f"Missing data for token at pos {pos}: LAB_{itemid}")
                continue
            # Pop all earlier-than-last_time entries (these indicate potential out-of-order if exist)
            popped_earlier = 0
            if last_time is not None:
                while times and times[0] < last_time:
                    times.pop(0)
                    popped_earlier += 1
                if popped_earlier > 0:
                    out_of_order += 1
                    examples.append(f"Out-of-order token at pos {pos}: LAB_{itemid} (skipped {popped_earlier} earlier event(s))")
            # Now match the next available time >= last_time
            if times:
                matched_time = times.pop(0)
                matched += 1
                last_time = matched_time
            else:
                # No remaining events for this itemid at/after last_time
                missing += 1
                examples.append(f"No remaining event >= prior time for token at pos {pos}: LAB_{itemid}")
        print(f"Matched tokens: {matched}")
        print(f"Out-of-order tokens: {out_of_order}")
        print(f"Tokens with no matching event: {missing}")
        if examples:
            print("\nExamples:")
            for msg in examples[:10]:
                print("  - " + msg)

    # Analyze each lab token
    print(f"\n📋 Detailed Analysis:")
    print("=" * 80)
    
    for i, (pos, itemid, quantile) in enumerate(lab_tokens):
        # Get actual results for this itemid
        item_data = actual_data.filter(pl.col("itemid") == itemid)
        
        status = "✅" if len(item_data) > 0 else "❌"
        quantile_str = f" (Q{quantile})" if quantile is not None else " (no quantile)"
        name = lab_names.get(itemid, "Unknown")
        
        print(f"{status} LAB_{itemid}{quantile_str} - {name}")
        
        if len(item_data) > 0:
            values = item_data["value"].drop_nulls()
            if len(values) > 0:
                print(f"    📊 Values: {len(values)} results")
                
                # Try to convert to numeric values, filtering out non-numeric ones
                try:
                    numeric_values = values.cast(pl.Float64, strict=False).drop_nulls()
                    if len(numeric_values) > 0:
                        print(f"    📈 Range: {numeric_values.min():.2f} - {numeric_values.max():.2f}")
                        print(f"    📊 Mean: {numeric_values.mean():.2f}")
                        
                        # Show first few values
                        sample_values = numeric_values.head(5).to_list()
                        print(f"    🔍 Sample: {sample_values}")
                    else:
                        print(f"    ⚠️  No numeric values found")
                        # Show raw values instead
                        sample_values = values.head(5).to_list()
                        print(f"    🔍 Raw sample: {sample_values}")
                except Exception as e:
                    print(f"    ⚠️  Could not convert to numeric: {e}")
                    # Show raw values instead
                    sample_values = values.head(5).to_list()
                    print(f"    🔍 Raw sample: {sample_values}")
            else:
                print(f"    ⚠️  No valid values found")
        else:
            print(f"    ❌ No data found for this itemid")
        
        print()
    
    # Validate ordering of LAB tokens against labevents charttime
    validate_lab_ordering(lab_tokens, actual_data)

    # Summary
    tokens_with_data = sum(1 for _, itemid, _ in lab_tokens 
                          if len(actual_data.filter(pl.col("itemid") == itemid)) > 0)
    
    print("=" * 80)
    print(f"📊 SUMMARY:")
    print(f"  ✅ Lab tokens with actual data: {tokens_with_data}/{len(lab_tokens)}")
    print(f"  ❌ Lab tokens without data: {len(lab_tokens) - tokens_with_data}/{len(lab_tokens)}")
    print(f"  📈 Total actual lab results: {len(actual_data)}")
    
    # Check for specific patterns
    print(f"\n🔍 PATTERN ANALYSIS:")
    
    # Count LAB_51222 occurrences
    lab_51222_count = sum(1 for _, itemid, _ in lab_tokens if itemid == 51222)
    lab_51222_data = actual_data.filter(pl.col("itemid") == 51222)
    
    print(f"  LAB_51222 appears {lab_51222_count} times in timeline")
    print(f"  LAB_51222 has {len(lab_51222_data)} actual results")
    
    if len(lab_51222_data) > 0:
        values = lab_51222_data["value"].drop_nulls()
        if len(values) > 0:
            try:
                numeric_values = values.cast(pl.Float64, strict=False).drop_nulls()
                if len(numeric_values) > 0:
                    print(f"  LAB_51222 value range: {numeric_values.min():.2f} - {numeric_values.max():.2f}")
                else:
                    print(f"  LAB_51222 has non-numeric values")
            except Exception as e:
                print(f"  LAB_51222 could not convert to numeric: {e}")

def main():
    """Run the validation"""
    validate_timeline(TIMELINE, HADM_ID, MIMIC_DATA_PATH)

if __name__ == "__main__":
    main()
