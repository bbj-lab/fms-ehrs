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
HADM_ID = 27880650

# Timeline to validate (replace with your actual timeline)
# TIMELINE = [
#     'TL_START', 'RACE_black/african_american', 'SEX_m', 'Q4', 'ADMN_ew_emer.', 
#     'PROC_0fb23zx', 'LAB_50971', 'Q2', 'Q3', 'LAB_51250', 'Q9', 'LAB_50983', 
#     'LAB_51657', 'Q9', 'Q9', 'Q9', 'LAB_50937', 'LAB_50947', 'Q7', 'LAB_51301', 
#     'LAB_50878', 'Q0', 'LAB_50902', 'Q9', 'Q0', 'LAB_51249', 'Q0', 'LAB_50882', 
#     'Q2', 'LAB_50947', 'Q9', 'LAB_51662', 'Q9', 'LAB_50943', 'Q9', 'LAB_51222', 
#     'LAB_51663', 'LAB_50912', 'Q9', 'LAB_51659', 'Q9', 'Q9', 'LAB_50940', 
#     'LAB_51006', 'Q0', 'LAB_50868', 'LAB_51265', 'Q2', 'LAB_52172', 'Q0', 
#     'LAB_51678', 'LAB_51248', 'Q0', 'LAB_50861', 'Q0', 'LAB_50934', 'Q0', 'Q0', 
#     'Q9', 'LAB_51237', 'LAB_50938', 'Q1', 'Q9', 'Q9', 'LAB_51279', 'Q0', 
#     'LAB_50934', 'Q5', 'LAB_51274', 'LAB_50954', 'LAB_50893', 'LAB_51221', 
#     'LAB_50931', 'LAB_50941', 'Q9', 'LAB_50954', 'Q0', 'Q9', 'Q0', 'LAB_50863', 
#     'LAB_51277', 'LAB_51658', 'LAB_51678', 'LAB_50885', 'Q9', 'Q9', 'LAB_51660', 
#     'Q9', 'LAB_50942', 'LAB_51476', 'LAB_51466', 'LAB_51508', 'LAB_51498', 'Q9', 
#     'LAB_51516', 'LAB_51484', 'Q0', 'Q9', 'LAB_51519', 'LAB_51493', 'LAB_52425', 
#     'LAB_51514', 'LAB_51506', 'LAB_51486', 'LAB_51492', 'Q9', 'LAB_51464', 
#     'LAB_51487', 'TEXT_DONE', 'Q5', 'LAB_51463', 'LAB_51491', 'LAB_51512', 
#     'LAB_51478', 'LAB_51839', 'Q6', 'Q3', 'LAB_51678', 'Q0', 'LAB_51248', 'Q0', 
#     'Q0', 'LAB_50902', 'Q0', 'Q9', 'LAB_50912', 'Q9', 'LAB_51250', 'LAB_50947', 
#     'Q2', 'Q9', 'LAB_51279', 'LAB_51222', 'Q0', 'LAB_51221', 'Q0', 'LAB_51301', 
#     'Q0', 'LAB_50931', 'LAB_51265', 'LAB_51274', 'Q9', 'Q9', 'LAB_50983', 'Q2', 
#     'LAB_50885', 'Q9', 'Q9', 'LAB_51277', 'Q8', 'LAB_50868', 'LAB_50971', 'Q9', 
#     'LAB_51237', 'LAB_50878', 'LAB_50882', 'Q0', 'LAB_50970', 'Q5', 'Q0', 'Q0', 
#     'LAB_50960', 'LAB_51249', 'Q6', 'LAB_50863', 'LAB_50861', 'Q0', 'LAB_52172', 
#     'Q0', 'Q9', 'LAB_50934', 'Q3', 'LAB_51006', 'LAB_50868', 'Q0', 'Q0', 'Q9', 
#     'LAB_50882', 'Q9', 'LAB_51248', 'Q0', 'LAB_50902', 'LAB_51250', 'LAB_51249', 
#     'LAB_50970', 'Q0', 'LAB_51006', 'Q9', 'LAB_50947', 'LAB_50912', 'Q9', 
#     'LAB_51678', 'Q0', 'Q0', 'LAB_52172', 'Q0', 'Q5', 'LAB_50971', 'Q9', 'Q3', 
#     'LAB_50861', 'Q0', 'LAB_50934', 'Q1', 'LAB_51279', 'Q2', 'LAB_51222', 
#     'LAB_50863', 'Q0', 'Q0', 'LAB_51277', 'LAB_51221', 'Q9', 'Q9', 'LAB_50960', 
#     'Q0', 'Q9', 'Q0', 'LAB_50885', 'LAB_50878', 'Q9', 'LAB_51301', 'Q0', 'Q2', 
#     'LAB_50983', 'Q0', 'LAB_50931', 'LAB_51265', 'LAB_51237', 'LAB_51274', 'Q9', 
#     'Q0', 'LAB_51301', 'LAB_51222', 'Q9', 'Q0', 'LAB_51277', 'Q0', 'LAB_51221', 
#     'LAB_51249', 'Q0', 'Q0', 'Q0', 'LAB_52172', 'LAB_51265', 'Q9', 'LAB_51279', 
#     'Q9', 'LAB_51250', 'Q9', 'LAB_51248', 'Q0', 'Q9', 'LAB_52172', 'Q0', 'Q9', 
#     'LAB_50882', 'Q0', 'LAB_51250', 'LAB_51265', 'LAB_51006', 'Q9', 'LAB_50878', 
#     'Q5', 'LAB_50902', 'LAB_50924', 'Q8', 'Q9', 'Q9', 'Q3', 'LAB_50998', 
#     'LAB_50971', 'LAB_50983', 'Q0', 'Q0', 'LAB_50863', 'Q0', 'Q0', 'Q9', 
#     'LAB_50885', 'LAB_50868', 'LAB_50912', 'LAB_51248', 'Q9', 'Q0', 'Q3', 
#     'LAB_50861', 'LAB_51678', 'Q0', 'LAB_51277', 'LAB_51283', 'Q9', 'Q6', 'Q9', 
#     'LAB_51279', 'LAB_50934', 'LAB_51249', 'Q0', 'Q1', 'LAB_50947', 'LAB_50953', 
#     'LAB_51301', 'Q0', 'Q9', 'Q0', 'LAB_50931', 'Q3', 'Q0', 'LAB_51221', 
#     'LAB_51282', 'LAB_51222', 'Q0', 'LAB_50952', 'DSCG_home', 'TL_END'
# ]
# TIMELINE = ['TL_START', 'RACE_black/african_american', 'SEX_m', 'Q4', 'ADMN_ew_emer.', 'PROC_0fb23zx', 'Q9', 'LAB_50940', 'Q7', 'LAB_50971', 'LAB_51660', 'Q9', 'Q8', 'LAB_51678', 'Q6', 'LAB_51678', 'Q6', 'Q4', 'LAB_51221', 'Q9', 'LAB_50943', 'LAB_50954', 'Q9', 'LAB_51301', 'LAB_51663', 'LAB_50893', 'LAB_50954', 'Q7', 'Q2', 'LAB_51006', 'LAB_50941', 'Q0', 'Q1', 'LAB_50983', 'Q1', 'LAB_50878', 'LAB_50938', 'LAB_50942', 'Q9', 'LAB_50947', 'Q9', 'LAB_50947', 'Q6', 'Q3', 'Q9', 'LAB_50937', 'LAB_52172', 'LAB_51249', 'LAB_51662', 'Q9', 'LAB_51237', 'Q7', 'LAB_50931', 'LAB_51265', 'Q8', 'Q8', 'Q6', 'LAB_51279', 'Q2', 'LAB_50882', 'LAB_50934', 'Q9', 'LAB_51658', 'LAB_50912', 'Q5', 'LAB_51222', 'LAB_50861', 'Q4', 'Q8', 'Q9', 'LAB_50863', 'Q1', 'Q9', 'LAB_50934', 'Q4', 'LAB_51659', 'LAB_51274', 'LAB_50868', 'LAB_51250', 'Q4', 'LAB_50902', 'LAB_51277', 'Q4', 'LAB_51248', 'Q0', 'Q6', 'LAB_50885', 'Q0', 'LAB_51657', 'LAB_51486', 'LAB_52425', 'LAB_51508', 'LAB_51519', 'Q1', 'LAB_51514', 'LAB_51476', 'Q3', 'LAB_51492', 'LAB_51484', 'Q9', 'LAB_51464', 'LAB_51466', 'LAB_51512', 'LAB_51493', 'TEXT_DONE', 'LAB_51478', 'LAB_51463', 'LAB_51491', 'LAB_51487', 'LAB_51516', 'LAB_51506', 'Q5', 'LAB_51498', 'Q9', 'LAB_51839', 'LAB_50960', 'Q8', 'LAB_51248', 'Q0', 'LAB_50971', 'Q6', 'Q1', 'LAB_50983', 'Q0', 'LAB_51250', 'Q4', 'LAB_51277', 'LAB_50885', 'Q5', 'Q4', 'Q3', 'LAB_50861', 'Q8', 'LAB_51265', 'Q0', 'LAB_51221', 'Q2', 'Q8', 'LAB_51249', 'LAB_50931', 'Q8', 'LAB_50934', 'Q3', 'LAB_51279', 'Q5', 'Q3', 'LAB_50868', 'Q8', 'Q3', 'LAB_50902', 'Q5', 'LAB_51301', 'LAB_50882', 'Q5', 'LAB_50970', 'LAB_51678', 'Q1', 'LAB_50878', 'Q8', 'LAB_50912', 'Q7', 'LAB_50863', 'Q8', 'LAB_52172', 'LAB_51237', 'Q0', 'LAB_51006', 'Q2', 'LAB_51274', 'Q8', 'Q9', 'LAB_50947', 'LAB_51222', 'LAB_51274', 'LAB_50912', 'Q0', 'Q2', 'LAB_50971', 'Q7', 'LAB_51221', 'Q1', 'LAB_51222', 'LAB_50861', 'LAB_51249', 'Q4', 'Q6', 'Q3', 'Q9', 'LAB_50947', 'LAB_50970', 'Q1', 'LAB_50983', 'Q7', 'Q6', 'LAB_50931', 'Q6', 'Q8', 'LAB_51237', 'LAB_51279', 'LAB_50934', 'Q1', 'Q3', 'Q9', 'Q8', 'LAB_50960', 'Q3', 'Q7', 'LAB_50863', 'LAB_51301', 'Q4', 'LAB_51006', 'Q1', 'Q4', 'LAB_50882', 'Q0', 'LAB_51248', 'Q9', 'LAB_51678', 'Q0', 'LAB_51250', 'LAB_50868', 'Q2', 'Q1', 'LAB_50878', 'LAB_52172', 'Q3', 'LAB_50902', 'LAB_50885', 'LAB_51265', 'LAB_51277', 'Q7', 'Q6', 'LAB_51277', 'LAB_51301', 'Q5', 'Q4', 'LAB_51222', 'LAB_51279', 'LAB_52172', 'Q0', 'LAB_51250', 'Q0', 'Q4', 'Q0', 'Q7', 'LAB_51265', 'LAB_51249', 'LAB_51221', 'Q3', 'LAB_51248', 'Q9', 'Q2', 'Q9', 'LAB_51283', 'Q9', 'LAB_51265', 'Q2', 'LAB_50998', 'Q5', 'LAB_50952', 'LAB_50971', 'LAB_50861', 'LAB_50882', 'Q2', 'LAB_50885', 'Q0', 'LAB_52172', 'Q3', 'LAB_50912', 'LAB_51301', 'Q3', 'Q5', 'LAB_51279', 'Q7', 'LAB_51277', 'LAB_50902', 'Q6', 'Q4', 'Q8', 'LAB_50868', 'Q0', 'LAB_50878', 'Q1', 'LAB_51678', 'Q0', 'Q1', 'LAB_51006', 'LAB_50863', 'Q7', 'Q1', 'LAB_50934', 'LAB_51250', 'Q0', 'LAB_50924', 'LAB_51222', 'Q6', 'LAB_50931', 'Q8', 'LAB_51248', 'LAB_51282', 'Q5', 'LAB_51249', 'Q0', 'LAB_50983', 'LAB_50947', 'Q5', 'Q0', 'Q2', 'LAB_50953', 'Q6', 'Q2', 'LAB_51221', 'Q5', 'DSCG_home', 'TL_END']
# TIMELINE = ['TL_START', 'RACE_black/african_american', 'SEX_m', 'Q4', 'ADMN_ew_emer.', 'PROC_0fb23zx', 'LAB_50943', 'LAB_50912', 'Q9', 'Q4', 'LAB_50868', 'Q6', 'LAB_51006', 'Q8', 'LAB_50931', 'Q8', 'Q3', 'LAB_51250', 'LAB_51248', 'LAB_51274', 'LAB_50861', 'LAB_51678', 'Q4', 'Q6', 'Q2', 'Q0', 'LAB_51678', 'Q7', 'LAB_50940', 'LAB_51657', 'Q9', 'Q6', 'LAB_50863', 'Q6', 'LAB_50983', 'Q1', 'LAB_51279', 'LAB_50885', 'LAB_50938', 'LAB_51265', 'LAB_51658', 'Q8', 'LAB_51222', 'LAB_50934', 'Q0', 'LAB_51221', 'LAB_51660', 'Q4', 'Q5', 'Q4', 'LAB_51659', 'Q9', 'LAB_51237', 'Q9', 'Q9', 'Q8', 'Q9', 'LAB_50971', 'Q0', 'Q9', 'LAB_51301', 'LAB_50902', 'Q1', 'LAB_52172', 'LAB_50934', 'LAB_50893', 'LAB_50942', 'Q6', 'LAB_50937', 'LAB_50941', 'LAB_51662', 'Q7', 'LAB_51249', 'Q1', 'Q2', 'LAB_50878', 'Q4', 'LAB_51277', 'Q9', 'Q9', 'LAB_51663', 'Q7', 'LAB_50882', 'Q9', 'LAB_50954', 'LAB_50947', 'Q9', 'LAB_50947', 'LAB_50954', 'LAB_51512', 'LAB_51492', 'Q3', 'LAB_51519', 'LAB_51484', 'LAB_51463', 'Q9', 'LAB_51498', 'Q5', 'LAB_51466', 'LAB_52425', 'LAB_51486', 'LAB_51516', 'LAB_51476', 'LAB_51491', 'TEXT_DONE', 'LAB_51493', 'LAB_51508', 'LAB_51478', 'LAB_51514', 'Q9', 'LAB_51506', 'LAB_51487', 'LAB_51464', 'Q1', 'LAB_51839', 'Q0', 'LAB_51279', 'Q3', 'LAB_50902', 'Q4', 'Q8', 'LAB_51250', 'Q0', 'LAB_50868', 'Q8', 'Q8', 'LAB_50931', 'Q3', 'LAB_51678', 'LAB_50971', 'Q2', 'Q8', 'LAB_51248', 'Q0', 'LAB_51237', 'Q8', 'LAB_50970', 'LAB_51265', 'LAB_50861', 'Q6', 'LAB_50960', 'LAB_52172', 'Q5', 'Q7', 'LAB_50863', 'Q5', 'LAB_50934', 'LAB_50885', 'LAB_50983', 'Q1', 'LAB_51006', 'LAB_51274', 'Q8', 'Q8', 'LAB_51249', 'LAB_51221', 'Q2', 'Q4', 'Q3', 'LAB_51222', 'LAB_51277', 'Q3', 'Q1', 'Q8', 'LAB_50878', 'Q0', 'LAB_50882', 'LAB_50947', 'LAB_50912', 'Q5', 'LAB_51301', 'Q9', 'Q5', 'Q7', 'Q1', 'LAB_50947', 'Q9', 'LAB_51006', 'LAB_50912', 'Q3', 'LAB_50861', 'LAB_51265', 'LAB_51301', 'LAB_50882', 'LAB_50878', 'Q8', 'Q2', 'LAB_51222', 'LAB_50960', 'Q4', 'Q1', 'LAB_51250', 'LAB_50971', 'Q6', 'LAB_50931', 'Q1', 'LAB_51221', 'Q0', 'Q7', 'LAB_51249', 'Q0', 'Q6', 'LAB_51274', 'Q9', 'LAB_50868', 'Q2', 'Q0', 'Q1', 'LAB_50983', 'LAB_50934', 'Q3', 'LAB_50970', 'Q9', 'Q8', 'LAB_51277', 'LAB_50902', 'LAB_50885', 'Q6', 'Q4', 'Q3', 'LAB_50863', 'LAB_52172', 'LAB_51279', 'LAB_51248', 'LAB_51678', 'Q4', 'Q3', 'Q1', 'LAB_51237', 'Q7', 'Q7', 'Q0', 'LAB_51279', 'Q9', 'Q4', 'Q5', 'Q0', 'LAB_51250', 'LAB_51277', 'Q7', 'LAB_51265', 'LAB_51249', 'Q4', 'LAB_51221', 'Q0', 'LAB_51301', 'Q3', 'Q6', 'LAB_51222', 'LAB_52172', 'LAB_51248', 'Q2', 'Q9', 'LAB_50931', 'Q5', 'Q8', 'LAB_50924', 'Q1', 'LAB_51006', 'LAB_50863', 'Q2', 'LAB_50885', 'Q7', 'LAB_50861', 'Q9', 'Q1', 'LAB_50934', 'LAB_50882', 'LAB_51265', 'Q0', 'LAB_51282', 'Q6', 'LAB_50868', 'Q8', 'Q6', 'Q1', 'LAB_50983', 'LAB_50971', 'Q5', 'LAB_50998', 'LAB_51279', 'Q3', 'LAB_51301', 'Q5', 'LAB_51221', 'LAB_51248', 'Q0', 'LAB_52172', 'Q7', 'LAB_50912', 'LAB_50952', 'LAB_51678', 'Q0', 'Q0', 'Q3', 'LAB_50878', 'Q5', 'Q2', 'Q2', 'Q0', 'Q5', 'LAB_50953', 'LAB_50947', 'Q6', 'LAB_51283', 'Q0', 'LAB_51250', 'LAB_50902', 'Q4', 'LAB_51277', 'Q2', 'LAB_51222', 'LAB_51249', 'DSCG_home', 'TL_END']
TIMELINE = ['TL_START', 'RACE_black/african_american', 'SEX_m', 'Q4', 'ADMN_ew_emer.', 'LAB_50924', 'Q8', 'LAB_50938', 'LAB_51493', 'Q5', 'LAB_50934', 'Q1', 'LAB_50934', 'Q4', 'LAB_50934', 'Q3', 'LAB_50934', 'Q1', 'LAB_50934', 'Q1', 'LAB_51221', 'Q4', 'LAB_51221', 'Q2', 'LAB_51221', 'Q1', 'LAB_51221', 'Q3', 'LAB_51221', 'Q2', 'LAB_52172', 'Q0', 'LAB_52172', 'Q0', 'LAB_52172', 'Q0', 'LAB_52172', 'Q0', 'LAB_52172', 'Q0', 'LAB_51006', 'Q2', 'LAB_51006', 'Q2', 'LAB_51006', 'Q1', 'LAB_51006', 'Q1', 'LAB_51463', 'LAB_51478', 'LAB_50893', 'Q6', 'LAB_51492', 'Q9', 'LAB_51519', 'LAB_50868', 'Q4', 'LAB_50868', 'Q8', 'LAB_50868', 'Q2', 'LAB_50868', 'Q8', 'LAB_50960', 'Q3', 'LAB_50960', 'Q3', 'LAB_51248', 'Q0', 'LAB_51248', 'Q0', 'LAB_51248', 'Q0', 'LAB_51248', 'Q0', 'LAB_51248', 'Q0', 'LAB_50970', 'Q5', 'LAB_50970', 'Q7', 'LAB_51279', 'Q6', 'LAB_51279', 'Q5', 'LAB_51279', 'Q4', 'LAB_51279', 'Q6', 'LAB_51279', 'Q5', 'LAB_51658', 'LAB_51282', 'Q0', 'LAB_51486', 'LAB_50943', 'LAB_51514', 'LAB_50861', 'Q4', 'LAB_50861', 'Q4', 'LAB_50861', 'Q3', 'LAB_50861', 'Q3', 'LAB_51464', 'LAB_50885', 'Q6', 'LAB_50885', 'Q5', 'LAB_50885', 'Q6', 'LAB_50885', 'Q2', 'LAB_50947', 'Q9', 'LAB_50947', 'Q9', 'LAB_50947', 'Q9', 'LAB_50947', 'Q9', 'LAB_50947', 'Q5', 'LAB_51512', 'LAB_51476', 'LAB_50902', 'Q3', 'LAB_50902', 'Q3', 'LAB_50902', 'Q3', 'LAB_50902', 'Q6', 'LAB_51498', 'Q9', 'LAB_51491', 'Q1', 'LAB_51660', 'Q9', 'LAB_51222', 'Q5', 'LAB_51222', 'Q3', 'LAB_51222', 'Q2', 'LAB_51222', 'Q4', 'LAB_51222', 'Q2', 'LAB_50952', 'Q5', 'LAB_50953', 'Q2', 'LAB_51237', 'Q9', 'LAB_51237', 'Q8', 'LAB_51237', 'Q8', 'LAB_50878', 'Q1', 'LAB_50878', 'Q1', 'LAB_50878', 'Q1', 'LAB_50878', 'Q1', 'LAB_51277', 'Q4', 'LAB_51277', 'Q4', 'LAB_51277', 'Q4', 'LAB_51277', 'Q4', 'LAB_51277', 'Q4', 'LAB_50971', 'Q6', 'LAB_50971', 'Q6', 'LAB_50971', 'Q7', 'LAB_50971', 'Q6', 'LAB_50882', 'Q2', 'LAB_50882', 'Q0', 'LAB_50882', 'Q3', 'LAB_50882', 'Q0', 'LAB_51657', 'Q9', 'LAB_51516', 'Q3', 'LAB_51662', 'Q9', 'LAB_51839', 'LAB_51659', 'Q9', 'LAB_51301', 'Q7', 'LAB_51301', 'Q5', 'LAB_51301', 'Q4', 'LAB_51301', 'Q5', 'LAB_51301', 'Q3', 'LAB_50931', 'Q8', 'LAB_50931', 'Q8', 'LAB_50931', 'Q6', 'LAB_50931', 'Q6', 'LAB_51663', 'Q9', 'LAB_50937', 'LAB_51274', 'Q9', 'LAB_51274', 'Q8', 'LAB_51274', 'Q8', 'LAB_51508', 'LAB_50998', 'Q2', 'LAB_50940', 'LAB_51466', 'LAB_51283', 'Q0', 'LAB_50912', 'Q8', 'LAB_50912', 'Q8', 'LAB_50912', 'Q7', 'LAB_50912', 'Q7', 'LAB_50863', 'Q7', 'LAB_50863', 'Q7', 'LAB_50863', 'Q7', 'LAB_50863', 'Q7', 'LAB_52425', 'TEXT_DONE', 'LAB_51484', 'LAB_50954', 'Q9', 'LAB_50954', 'Q9', 'LAB_51487', 'LAB_50942', 'LAB_51250', 'Q0', 'LAB_51250', 'Q0', 'LAB_51250', 'Q0', 'LAB_51250', 'Q0', 'LAB_51250', 'Q0', 'LAB_50983', 'Q1', 'LAB_50983', 'Q1', 'LAB_50983', 'Q1', 'LAB_50983', 'Q5', 'LAB_51506', 'LAB_51678', 'Q6', 'LAB_51678', 'Q8', 'LAB_51678', 'Q8', 'LAB_51678', 'Q9', 'LAB_51678', 'Q9', 'LAB_50941', 'LAB_51249', 'Q7', 'LAB_51249', 'Q8', 'LAB_51249', 'Q6', 'LAB_51249', 'Q7', 'LAB_51249', 'Q5', 'LAB_51265', 'Q8', 'LAB_51265', 'Q8', 'LAB_51265', 'Q9', 'LAB_51265', 'Q9', 'LAB_51265', 'Q9', 'PROC_0fb23zx', 'DSCG_home', 'TL_END']

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
