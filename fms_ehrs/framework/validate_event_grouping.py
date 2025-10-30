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
#  ['TL_START_2137-12-11T14:57:00', 'RACE_white_2137-12-11T14:57:00', 'SEX_f_2137-12-11T14:57:00', 'Q9_2137-12-11T14:57:00', 'ADMN_direct_observation_2137-12-11T14:57:00', 'LAB_50912_2137-12-12T06:25:00', 'LAB_50947_2137-12-12T06:25:00', 'LAB_51279_2137-12-12T06:25:00', 'LAB_50960_2137-12-12T06:25:00', 'LAB_50971_2137-12-12T06:25:00', 'LAB_50994_2137-12-12T06:25:00', 'LAB_51222_2137-12-12T06:25:00', 'LAB_50970_2137-12-12T06:25:00', 'LAB_50893_2137-12-12T06:25:00', 'LAB_51678_2137-12-12T06:25:00', 'LAB_51250_2137-12-12T06:25:00', 'LAB_51249_2137-12-12T06:25:00', 'LAB_50993_2137-12-12T06:25:00', 'LAB_50902_2137-12-12T06:25:00', 'LAB_50931_2137-12-12T06:25:00', 'LAB_51221_2137-12-12T06:25:00', 'LAB_50934_2137-12-12T06:25:00', 'LAB_51006_2137-12-12T06:25:00', 'LAB_51301_2137-12-12T06:25:00', 'LAB_50882_2137-12-12T06:25:00', 'LAB_51277_2137-12-12T06:25:00', 'LAB_50868_2137-12-12T06:25:00', 'LAB_51248_2137-12-12T06:25:00', 'LAB_52172_2137-12-12T06:25:00', 'LAB_51265_2137-12-12T06:25:00', 'LAB_50983_2137-12-12T06:25:00', 'DSCG_UNKNOWN_2137-12-12T17:01:00', 'TL_END_2137-12-12T17:01:00']
# Example timeline (hadm_id: 28301173): 
#  ['TL_START_2197-04-08T19:37:00', 'RACE_black/african_american_2197-04-08T19:37:00', 'SEX_f_2197-04-08T19:37:00', 'Q2_2197-04-08T19:37:00', 'ADMN_urgent_2197-04-08T19:37:00', 'DSCG_home_2197-04-15T12:01:00', 'TL_END_2197-04-15T12:01:00']
# Example timeline (hadm_id: 27773809): 
TIMELINE = ['TL_START_2124-06-27T15:04:00', 'RACE_white_2124-06-27T15:04:00', 'SEX_m_2124-06-27T15:04:00', 'Q2_2124-06-27T15:04:00', 'ADMN_observation_admit_2124-06-27T15:04:00', 'PROC_009u3zx_2124-06-26T00:00:00', 'LAB_50861_2124-06-26T11:00:00', 'LAB_50862_2124-06-26T11:00:00', 'LAB_50863_2124-06-26T11:00:00', 'LAB_50868_2124-06-26T11:00:00', 'LAB_50878_2124-06-26T11:00:00', 'LAB_50882_2124-06-26T11:00:00', 'LAB_50885_2124-06-26T11:00:00', 'LAB_50902_2124-06-26T11:00:00', 'LAB_50912_2124-06-26T11:00:00', 'LAB_50920_2124-06-26T11:00:00', 'LAB_50931_2124-06-26T11:00:00', 'LAB_50934_2124-06-26T11:00:00', 'LAB_50947_2124-06-26T11:00:00', 'LAB_50971_2124-06-26T11:00:00', 'LAB_50983_2124-06-26T11:00:00', 'LAB_51006_2124-06-26T11:00:00', 'LAB_51678_2124-06-26T11:00:00', 'LAB_51688_2124-06-26T11:00:00', 'LAB_51689_2124-06-26T11:00:00', 'LAB_50933_2124-06-26T11:00:00', 'LAB_50955_2124-06-26T11:00:00', 'LAB_50887_2124-06-26T11:00:00', 'LAB_51133_2124-06-26T11:00:00', 'LAB_51146_2124-06-26T11:00:00', 'LAB_51200_2124-06-26T11:00:00', 'LAB_51221_2124-06-26T11:00:00', 'LAB_51222_2124-06-26T11:00:00', 'LAB_51244_2124-06-26T11:00:00', 'LAB_51248_2124-06-26T11:00:00', 'LAB_51249_2124-06-26T11:00:00', 'LAB_51250_2124-06-26T11:00:00', 'LAB_51254_2124-06-26T11:00:00', 'LAB_51256_2124-06-26T11:00:00', 'LAB_51265_2124-06-26T11:00:00', 'LAB_51277_2124-06-26T11:00:00', 'LAB_51279_2124-06-26T11:00:00', 'LAB_51301_2124-06-26T11:00:00', 'LAB_52069_2124-06-26T11:00:00', 'LAB_52073_2124-06-26T11:00:00', 'LAB_52074_2124-06-26T11:00:00', 'LAB_52075_2124-06-26T11:00:00', 'LAB_52135_2124-06-26T11:00:00', 'LAB_52172_2124-06-26T11:00:00', 'LAB_50813_2124-06-26T11:49:00', 'LAB_50825_2124-06-26T11:49:00', 'LAB_52033_2124-06-26T11:49:00', 'TEXT_VEN_2124-06-26T11:49:00', 'LAB_50813_2124-06-26T14:47:00', 'LAB_52264_2124-06-26T15:40:00', 'LAB_52272_2124-06-26T15:40:00', 'LAB_52278_2124-06-26T15:40:00', 'LAB_52281_2124-06-26T15:40:00', 'LAB_52285_2124-06-26T15:40:00', 'LAB_52286_2124-06-26T15:40:00', 'LAB_51790_2124-06-26T15:40:00', 'LAB_51802_2124-06-26T15:40:00', 'LAB_52225_2124-06-26T15:40:00', 'LAB_52264_2124-06-26T15:40:00', 'LAB_52272_2124-06-26T15:40:00', 'LAB_52278_2124-06-26T15:40:00', 'LAB_52281_2124-06-26T15:40:00', 'LAB_52285_2124-06-26T15:40:00', 'LAB_52286_2124-06-26T15:40:00', 'LAB_51564_2124-06-28T07:20:00', 'LAB_53153_2124-06-28T07:20:00', 'LAB_51137_2124-06-28T07:20:00', 'LAB_51221_2124-06-28T07:20:00', 'LAB_51222_2124-06-28T07:20:00', 'LAB_51233_2124-06-28T07:20:00', 'LAB_51246_2124-06-28T07:20:00', 'LAB_51248_2124-06-28T07:20:00', 'LAB_51249_2124-06-28T07:20:00', 'LAB_51250_2124-06-28T07:20:00', 'LAB_51252_2124-06-28T07:20:00', 'LAB_51265_2124-06-28T07:20:00', 'LAB_51268_2124-06-28T07:20:00', 'LAB_51277_2124-06-28T07:20:00', 'LAB_51279_2124-06-28T07:20:00', 'LAB_51296_2124-06-28T07:20:00', 'LAB_51301_2124-06-28T07:20:00', 'LAB_52171_2124-06-28T07:20:00', 'LAB_52172_2124-06-28T07:20:00', 'LAB_50853_2124-06-28T07:20:00', 'LAB_50861_2124-06-28T07:20:00', 'LAB_50863_2124-06-28T07:20:00', 'LAB_50868_2124-06-28T07:20:00', 'LAB_50878_2124-06-28T07:20:00', 'LAB_50882_2124-06-28T07:20:00', 'LAB_50885_2124-06-28T07:20:00', 'LAB_50893_2124-06-28T07:20:00', 'LAB_50902_2124-06-28T07:20:00', 'LAB_50912_2124-06-28T07:20:00', 'LAB_50924_2124-06-28T07:20:00', 'LAB_50925_2124-06-28T07:20:00', 'LAB_50931_2124-06-28T07:20:00', 'LAB_50934_2124-06-28T07:20:00', 'LAB_50947_2124-06-28T07:20:00', 'LAB_50952_2124-06-28T07:20:00', 'LAB_50953_2124-06-28T07:20:00', 'LAB_50954_2124-06-28T07:20:00', 'LAB_50960_2124-06-28T07:20:00', 'LAB_50970_2124-06-28T07:20:00', 'LAB_50971_2124-06-28T07:20:00', 'LAB_50983_2124-06-28T07:20:00', 'LAB_50993_2124-06-28T07:20:00', 'LAB_50998_2124-06-28T07:20:00', 'LAB_51006_2124-06-28T07:20:00', 'LAB_51010_2124-06-28T07:20:00', 'LAB_51678_2124-06-28T07:20:00', 'LAB_51688_2124-06-28T07:20:00', 'LAB_51689_2124-06-28T07:20:00', 'LAB_51150_2124-06-28T16:45:00', 'TEXT_NEGATIVE_2124-06-28T16:45:00', 'LAB_51748_2124-06-28T16:45:00', 'LAB_51749_2124-06-28T16:45:00', 'LAB_51133_2124-06-29T03:30:00', 'LAB_51146_2124-06-29T03:30:00', 'LAB_51200_2124-06-29T03:30:00', 'LAB_51221_2124-06-29T03:30:00', 'LAB_51222_2124-06-29T03:30:00', 'LAB_51244_2124-06-29T03:30:00', 'LAB_51248_2124-06-29T03:30:00', 'LAB_51249_2124-06-29T03:30:00', 'LAB_51250_2124-06-29T03:30:00', 'LAB_51254_2124-06-29T03:30:00', 'LAB_51256_2124-06-29T03:30:00', 'LAB_51265_2124-06-29T03:30:00', 'LAB_51277_2124-06-29T03:30:00', 'LAB_51279_2124-06-29T03:30:00', 'LAB_51301_2124-06-29T03:30:00', 'LAB_52069_2124-06-29T03:30:00', 'LAB_52073_2124-06-29T03:30:00', 'LAB_52074_2124-06-29T03:30:00', 'LAB_52075_2124-06-29T03:30:00', 'LAB_52135_2124-06-29T03:30:00', 'LAB_52172_2124-06-29T03:30:00', 'LAB_50868_2124-06-29T03:30:00', 'LAB_50882_2124-06-29T03:30:00', 'LAB_50885_2124-06-29T03:30:00', 'LAB_50893_2124-06-29T03:30:00', 'LAB_50902_2124-06-29T03:30:00', 'LAB_50912_2124-06-29T03:30:00', 'LAB_50931_2124-06-29T03:30:00', 'LAB_50934_2124-06-29T03:30:00', 'LAB_50947_2124-06-29T03:30:00', 'LAB_50949_2124-06-29T03:30:00', 'LAB_50960_2124-06-29T03:30:00', 'LAB_50970_2124-06-29T03:30:00', 'LAB_50971_2124-06-29T03:30:00', 'LAB_50983_2124-06-29T03:30:00', 'LAB_50996_2124-06-29T03:30:00', 'LAB_51006_2124-06-29T03:30:00', 'LAB_51678_2124-06-29T03:30:00', 'LAB_51214_2124-06-29T03:30:00', 'LAB_51237_2124-06-29T03:30:00', 'LAB_51274_2124-06-29T03:30:00', 'LAB_51275_2124-06-29T03:30:00', 'LAB_51133_2124-06-30T07:20:00', 'LAB_51146_2124-06-30T07:20:00', 'LAB_51200_2124-06-30T07:20:00', 'LAB_51221_2124-06-30T07:20:00', 'LAB_51222_2124-06-30T07:20:00', 'LAB_51244_2124-06-30T07:20:00', 'LAB_51248_2124-06-30T07:20:00', 'LAB_51249_2124-06-30T07:20:00', 'LAB_51250_2124-06-30T07:20:00', 'LAB_51254_2124-06-30T07:20:00', 'LAB_51256_2124-06-30T07:20:00', 'LAB_51265_2124-06-30T07:20:00', 'LAB_51277_2124-06-30T07:20:00', 'LAB_51279_2124-06-30T07:20:00', 'LAB_51301_2124-06-30T07:20:00', 'LAB_52069_2124-06-30T07:20:00', 'LAB_52073_2124-06-30T07:20:00', 'LAB_52074_2124-06-30T07:20:00', 'LAB_52075_2124-06-30T07:20:00', 'LAB_52135_2124-06-30T07:20:00', 'LAB_52172_2124-06-30T07:20:00', 'LAB_50868_2124-06-30T07:20:00', 'LAB_50882_2124-06-30T07:20:00', 'LAB_50902_2124-06-30T07:20:00', 'LAB_50912_2124-06-30T07:20:00', 'LAB_50931_2124-06-30T07:20:00', 'LAB_50934_2124-06-30T07:20:00', 'LAB_50947_2124-06-30T07:20:00', 'LAB_50971_2124-06-30T07:20:00', 'LAB_50983_2124-06-30T07:20:00', 'LAB_51006_2124-06-30T07:20:00', 'LAB_51678_2124-06-30T07:20:00', 'LAB_51150_2124-06-30T15:05:00', 'TEXT_NEGATIVE_2124-06-30T15:05:00', 'LAB_51221_2124-06-30T20:09:00', 'LAB_51222_2124-06-30T20:09:00', 'LAB_50868_2124-07-01T09:10:00', 'LAB_50882_2124-07-01T09:10:00', 'LAB_50893_2124-07-01T09:10:00', 'LAB_50902_2124-07-01T09:10:00', 'LAB_50912_2124-07-01T09:10:00', 'LAB_50931_2124-07-01T09:10:00', 'LAB_50934_2124-07-01T09:10:00', 'LAB_50947_2124-07-01T09:10:00', 'LAB_50960_2124-07-01T09:10:00', 'LAB_50970_2124-07-01T09:10:00', 'LAB_50971_2124-07-01T09:10:00', 'LAB_50983_2124-07-01T09:10:00', 'LAB_51006_2124-07-01T09:10:00', 'LAB_51678_2124-07-01T09:10:00', 'LAB_51221_2124-07-01T09:10:00', 'LAB_51222_2124-07-01T09:10:00', 'LAB_51248_2124-07-01T09:10:00', 'LAB_51249_2124-07-01T09:10:00', 'LAB_51250_2124-07-01T09:10:00', 'LAB_51265_2124-07-01T09:10:00', 'LAB_51277_2124-07-01T09:10:00', 'LAB_51279_2124-07-01T09:10:00', 'LAB_51301_2124-07-01T09:10:00', 'LAB_52172_2124-07-01T09:10:00', 'LAB_51221_2124-07-02T05:45:00', 'LAB_51222_2124-07-02T05:45:00', 'LAB_51248_2124-07-02T05:45:00', 'LAB_51249_2124-07-02T05:45:00', 'LAB_51250_2124-07-02T05:45:00', 'LAB_51265_2124-07-02T05:45:00', 'LAB_51277_2124-07-02T05:45:00', 'LAB_51279_2124-07-02T05:45:00', 'LAB_51301_2124-07-02T05:45:00', 'LAB_52172_2124-07-02T05:45:00', 'LAB_50868_2124-07-02T05:45:00', 'LAB_50882_2124-07-02T05:45:00', 'LAB_50885_2124-07-02T05:45:00', 'LAB_50893_2124-07-02T05:45:00', 'LAB_50902_2124-07-02T05:45:00', 'LAB_50912_2124-07-02T05:45:00', 'LAB_50931_2124-07-02T05:45:00', 'LAB_50934_2124-07-02T05:45:00', 'LAB_50935_2124-07-02T05:45:00', 'LAB_50947_2124-07-02T05:45:00', 'LAB_50954_2124-07-02T05:45:00', 'LAB_50960_2124-07-02T05:45:00', 'LAB_50970_2124-07-02T05:45:00', 'LAB_50971_2124-07-02T05:45:00', 'LAB_50983_2124-07-02T05:45:00', 'LAB_51006_2124-07-02T05:45:00', 'LAB_51678_2124-07-02T05:45:00', 'LAB_51222_2124-07-02T21:50:00', 'PROC_0db78zx_2124-07-03T00:00:00', 'PROC_0dbp8zx_2124-07-03T00:00:00', 'LAB_51221_2124-07-03T05:20:00', 'LAB_51222_2124-07-03T05:20:00', 'LAB_51248_2124-07-03T05:20:00', 'LAB_51249_2124-07-03T05:20:00', 'LAB_51250_2124-07-03T05:20:00', 'LAB_51265_2124-07-03T05:20:00', 'LAB_51277_2124-07-03T05:20:00', 'LAB_51279_2124-07-03T05:20:00', 'LAB_51301_2124-07-03T05:20:00', 'LAB_52172_2124-07-03T05:20:00', 'LAB_50868_2124-07-03T05:20:00', 'LAB_50882_2124-07-03T05:20:00', 'LAB_50893_2124-07-03T05:20:00', 'LAB_50902_2124-07-03T05:20:00', 'LAB_50912_2124-07-03T05:20:00', 'LAB_50931_2124-07-03T05:20:00', 'LAB_50934_2124-07-03T05:20:00', 'LAB_50947_2124-07-03T05:20:00', 'LAB_50960_2124-07-03T05:20:00', 'LAB_50970_2124-07-03T05:20:00', 'LAB_50971_2124-07-03T05:20:00', 'LAB_50983_2124-07-03T05:20:00', 'LAB_51006_2124-07-03T05:20:00', 'LAB_51678_2124-07-03T05:20:00', 'LAB_50900_2124-07-04T05:50:00', 'LAB_50868_2124-07-04T09:45:00', 'LAB_50882_2124-07-04T09:45:00', 'LAB_50893_2124-07-04T09:45:00', 'LAB_50902_2124-07-04T09:45:00', 'LAB_50912_2124-07-04T09:45:00', 'LAB_50920_2124-07-04T09:45:00', 'LAB_50931_2124-07-04T09:45:00', 'LAB_50934_2124-07-04T09:45:00', 'LAB_50947_2124-07-04T09:45:00', 'LAB_50960_2124-07-04T09:45:00', 'LAB_50970_2124-07-04T09:45:00', 'LAB_50971_2124-07-04T09:45:00', 'LAB_50983_2124-07-04T09:45:00', 'LAB_51006_2124-07-04T09:45:00', 'LAB_51678_2124-07-04T09:45:00', 'LAB_51221_2124-07-04T09:45:00', 'LAB_51222_2124-07-04T09:45:00', 'LAB_51248_2124-07-04T09:45:00', 'LAB_51249_2124-07-04T09:45:00', 'LAB_51250_2124-07-04T09:45:00', 'LAB_51265_2124-07-04T09:45:00', 'LAB_51277_2124-07-04T09:45:00', 'LAB_51279_2124-07-04T09:45:00', 'LAB_51301_2124-07-04T09:45:00', 'LAB_52172_2124-07-04T09:45:00', 'LAB_50868_2124-07-05T06:00:00', 'LAB_50882_2124-07-05T06:00:00', 'LAB_50893_2124-07-05T06:00:00', 'LAB_50902_2124-07-05T06:00:00', 'LAB_50912_2124-07-05T06:00:00', 'LAB_50931_2124-07-05T06:00:00', 'LAB_50934_2124-07-05T06:00:00', 'LAB_50947_2124-07-05T06:00:00', 'LAB_50960_2124-07-05T06:00:00', 'LAB_50970_2124-07-05T06:00:00', 'LAB_50971_2124-07-05T06:00:00', 'LAB_50983_2124-07-05T06:00:00', 'LAB_51006_2124-07-05T06:00:00', 'LAB_51678_2124-07-05T06:00:00', 'LAB_51221_2124-07-05T06:00:00', 'LAB_51222_2124-07-05T06:00:00', 'LAB_51248_2124-07-05T06:00:00', 'LAB_51249_2124-07-05T06:00:00', 'LAB_51250_2124-07-05T06:00:00', 'LAB_51265_2124-07-05T06:00:00', 'LAB_51277_2124-07-05T06:00:00', 'LAB_51279_2124-07-05T06:00:00', 'LAB_51301_2124-07-05T06:00:00', 'LAB_52172_2124-07-05T06:00:00', 'DSCG_home_2124-07-05T17:50:00', 'TL_END_2124-07-05T17:50:00']
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
    # Match code tokens: PREFIX_CODE (e.g., LAB_50934, VTL_123, RACE_white, SEX_m)
    elif re.match(r'^[A-Z]+_[A-Za-z0-9]+', base_token):
        token_type = "code"
    
    return Token(
        position=position,
        token=full_token,
        base=base_token,
        token_type=token_type
    )

# =============================================================================
# SECTION DETECTION (PREFIX / EVENTS / SUFFIX)
# =============================================================================

def compute_event_section_mask(tokens: List[Token]) -> List[bool]:
    """
    Return a boolean mask of same length as tokens indicating positions that
    are in the clinical events section. We treat:
      - Prefix tokens: TL_START, RACE_*, SEX_*, ADMN_*, Q*
      - Suffix tokens: DSCG_*, TL_END
      - Everything else (codes like LAB_*, PROC_*, OMR_*, VTL_*, or any other
        UPPER_lower style code not in the prefix/suffix sets) as events.
    Once the first events code appears, we consider subsequent positions as
    events until a suffix token is encountered.
    """
    prefix_starts = ("RACE_", "SEX_", "ADMN_")
    suffix_starts = ("DSCG_",)

    in_events = False
    mask: List[bool] = []
    for t in tokens:
        # TL_START keeps us in prefix
        if t.base == "TL_START":
            mask.append(False)
            continue
        # TL_END is suffix; after this, not events
        if t.base == "TL_END" or any(t.base.startswith(s) for s in suffix_starts):
            in_events = False
            mask.append(False)
            continue
        # Quantiles are prefix unless we've already entered events
        if t.token_type == "quantile" and not in_events:
            mask.append(False)
            continue
        # Known prefix codes keep us in prefix until we hit a non-prefix code
        if t.token_type == "code" and not in_events and any(t.base.startswith(p) for p in prefix_starts):
            mask.append(False)
            continue
        # Any other code transitions us into events
        if t.token_type == "code" and not in_events:
            in_events = True
            mask.append(True)
            continue
        # If already in events, remain so unless suffix encountered
        mask.append(in_events)
    return mask

# =============================================================================
# VALIDATION
# =============================================================================

def validate_quantile_positions(tokens: List[Token]) -> List[dict]:
    """
    Rule: Every QUANTILE must be immediately after a CODE (events section only).
    QUANTILEs are optional; absence is fine.
    Prefix section exemption: quantiles that appear before the first clinical
    event code (e.g., LAB_/PROC_/VTL_/OMR_/MICRO_) are allowed standalone.
    """
    errors = []
    events_mask = compute_event_section_mask(tokens)

    for i, t in enumerate(tokens):
        if t.token_type == "quantile":
            # Only enforce inside the events section
            if not events_mask[i]:
                continue
            if i == 0 or tokens[i - 1].token_type != "code":
                errors.append({
                    'position': max(0, i - 1),
                    'error': 'QUANTILE_NOT_AFTER_CODE',
                    'pattern': '...-QUANTILE (prev not CODE)',
                    'tokens': tokens[max(0, i - 1): i + 1]
                })
    return errors

def validate_text_positions(tokens: List[Token]) -> List[dict]:
    """
    Rule: Every TEXT must be immediately after a CODE or a QUANTILE.
    TEXT is optional; absence is fine.
    """
    errors = []
    events_mask = compute_event_section_mask(tokens)
    for i, t in enumerate(tokens):
        if t.token_type == "text":
            # Only enforce inside the events section
            if not events_mask[i]:
                continue
            if i == 0 or tokens[i - 1].token_type not in ["code", "quantile"]:
                errors.append({
                    'position': max(0, i - 1),
                    'error': 'TEXT_NOT_AFTER_CODE_OR_QUANTILE',
                    'pattern': '...-TEXT (prev not CODE/QUANTILE)',
                    'tokens': tokens[max(0, i - 1): i + 1]
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

