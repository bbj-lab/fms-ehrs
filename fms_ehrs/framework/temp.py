

from asyncio.windows_events import NULL

event_table_schema = pa.DataFrameSchema(
    {
        "hadm_id": pa.Column(str, nullable=False),
        "subject_id": pa.Column(str, nullable=False),
        "time": pa.Column(pd.DatetimeTZDtype(unit="ms", tz="UTC"), nullable=False),
        "code": pa.Column(str, nullable=False),
        "numeric_value": pa.Column(float, nullable=True),
        "text_value": pa.Column(str, nullable=True)    
    },  
    strict=True,
)

prefix_table_schema = pa.DataFrameSchema(
    {
        "hadm_id": pa.Column(str, nullable=False),
        "subject_id": pa.Column(str, nullable=False),
        "admission_time": pa.Column(pd.DatetimeTZDtype(unit="ms", tz="UTC"), nullable=False),
        "race": pa.Column(str, nullable=True),
        "ethnicity": pa.Column(str, nullable=True),
        "sex": pa.Column(str, nullable=True),
        "age_at_admission": pa.Column(str, nullable=True),
        "admission_type": pa.Column(str, nullable=True),
    },  
    strict=True,
)

suffix_table_schema = pa.DataFrameSchema(
    {
        "hadm_id": pa.Column(str, nullable=False),
        "subject_id": pa.Column(str, nullable=False),
        "discharge_time": pa.Column(pd.DatetimeTZDtype(unit="ms", tz="UTC"), nullable=False),
        "discharge_category": pa.Column(str, nullable=True)
    },  
    strict=True,
)