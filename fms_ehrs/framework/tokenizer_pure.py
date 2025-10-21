#!/usr/bin/env python3

"""
A generic tokenizer that can tokenize any data as long as it's provided in a specific format.
This separates the tokenization process from data processing.
"""

import logging
import os
import pathlib
import typing

import numpy as np
import polars as pl

from fms_ehrs.framework.tokenizer0 import BaseTokenizer, summarize

Frame: typing.TypeAlias = pl.DataFrame | pl.LazyFrame
Pathlike: typing.TypeAlias = pathlib.PurePath | str | os.PathLike


class PureTokenizer(BaseTokenizer):
    """
    A generic tokenizer that works with any data processor.
    Separates tokenization logic from data processing.
    """
    
    def __init__(
        self,
        *,
        vocab_path: Pathlike = None,
        max_padded_len: int = None,
        quantizer: typing.Literal["deciles", "sigmas"] = "deciles",
        include_time_spacing_tokens: bool = False,
        cut_at_24h: bool = False,
        day_stay_filter: bool = False,
    ):
        """
        Initialize the pure tokenizer.
        
        Args:
            vocab_path: Path to existing vocabulary file (for inference mode)
            max_padded_len: Maximum length for padding/truncation
            quantizer: Quantization method for numerical values
            include_time_spacing_tokens: Whether to include time spacing tokens
            cut_at_24h: Whether to cut timelines at 24 hours
            day_stay_filter: Whether to filter for day stays only
        """
        super().__init__(
            data_dir=pathlib.Path("."),  # Dummy path, not used in pure tokenizer
            vocab_path=vocab_path,
            max_padded_len=max_padded_len,
            quantizer=quantizer,
            include_time_spacing_tokens=include_time_spacing_tokens,
        )
        self.cut_at_24h = cut_at_24h
        self.day_stay_filter = day_stay_filter
    
    def get_tokens_timelines(self, data_processor) -> Frame:
        """
        Generate tokenized timelines by combining prefix, events, and suffix tokens.
        
        Args:
            data_processor: An object with methods get_prefix_tokens(), get_events(), get_suffix_tokens()
        """
        # Get all components from the data processor
        prefix_tokens = data_processor.get_prefix_tokens()
        events = data_processor.get_events()
        suffix_tokens = data_processor.get_suffix_tokens()
        
        # Combine all components
        tt = (
            prefix_tokens
            .join(
                events,
                on="hospitalization_id",
                how="left",
                validate="1:1",
            )
            .join(
                suffix_tokens,
                on="hospitalization_id",
                how="left",
                validate="1:1",
            )
            .with_columns(
                tokens=pl.concat_list("prefix_tokens", "event_tokens", "suffix_tokens"),
                times=pl.concat_list("prefix_times", "event_times", "suffix_times"),
            )
            .select("hospitalization_id", "tokens", "times")
            .sort(by="hospitalization_id")
        )
        
        # Apply filters
        if self.day_stay_filter:
            tt = tt.filter(
                (pl.col("times").list.get(-1) - pl.col("times").list.get(0))
                >= pl.duration(days=1)
            )
        
        if self.cut_at_24h:
            tt = super().cut_at_time(tt)
        
        return tt.collect()
    
    def process_categorical_value(self, df: Frame, category_col: str, value_col: str, 
                                label: str, time_col: str = "event_time") -> Frame:
        """
        Process a dataframe with categorical values and numerical values.
        This is a generic version of the category-value processing logic.
        """
        # Group by category and process each group
        return pl.concat(
            self._process_single_category(x, label, time_col) 
            for x in df.partition_by(category_col)
        )
    
    def _process_single_category(self, x: Frame, label: str, time_col: str) -> Frame:
        """
        Process a single category group.
        """
        v = x.select("value").to_numpy().ravel()
        c = x.select("category").row(0)[0]
        self.set_quants(v=v, c=c, label=label)
        
        return (
            x.with_columns(
                token=pl.lit(self.vocab(f"{label}_{c}")).cast(pl.Int64),
                token_quantile=self.get_quants(v=v, c=c, label=label),
            )
            .filter(~pl.col("token").is_in([self.vocab(None), self.vocab("nan")]))
            .filter(
                ~pl.col("token_quantile").is_in([self.vocab(None), self.vocab("nan")])
            )
            .with_columns(
                tokens=pl.concat_list("token", "token_quantile").cast(
                    pl.List(pl.Int64)
                ),
                times=pl.concat_list(time_col, time_col).cast(
                    pl.List(pl.Datetime(time_unit="ms"))
                ),
            )
        )
    
    def process_categorical_only(self, df: Frame, category_col: str, label: str, 
                               time_col: str = "event_time") -> Frame:
        """
        Process a dataframe with only categorical values (no numerical values).
        """
        return df.select(
            "hospitalization_id",
            pl.col(time_col).cast(pl.Datetime(time_unit="ms")).alias("event_time"),
            pl.concat_list(
                [
                    pl.col(cat)
                    .str.to_lowercase()
                    .str.replace_all(" ", "_")
                    .str.strip_chars(".")
                    .map_elements(
                        lambda x, prefix=label: self.vocab(f"{prefix}_{x}"),
                        return_dtype=pl.Int64,
                        skip_nulls=False,
                    )
                    for cat in [category_col]
                ]
            ).alias("tokens"),
            pl.concat_list(
                [pl.col(time_col).cast(pl.Datetime("ms"))]
            ).alias("times"),
        ).collect()
    
    def create_timeline_tokens(self, df: Frame, token_columns: list[str], 
                             time_col: str, label: str = None) -> Frame:
        """
        Create timeline tokens from a dataframe with multiple token columns.
        
        Args:
            df: Input dataframe
            token_columns: List of columns to convert to tokens
            time_col: Column containing timestamps
            label: Optional prefix for tokens
        """
        return df.select(
            "hospitalization_id",
            pl.col(time_col).cast(pl.Datetime(time_unit="ms")).alias("event_time"),
            pl.concat_list(
                [
                    (
                        pl.col(col)
                        .str.to_lowercase()
                        .str.replace_all(" ", "_")
                        .map_elements(
                            lambda x, prefix=label: self.vocab(f"{prefix}_{x}"),
                            return_dtype=pl.Int64,
                            skip_nulls=False,
                        )
                        if not col.startswith("quantized")
                        else pl.col(col)
                    )
                    for col in token_columns
                ]
            ).alias("tokens"),
            pl.concat_list(
                [pl.col(time_col).cast(pl.Datetime(time_unit="ms"))]
                * len(token_columns)
            ).alias("times"),
        )


def create_timeline_summary(tokenizer: PureTokenizer, tokens_timelines: Frame, 
                          logger: logging.Logger = None) -> None:
    """
    Create a summary of the tokenized timelines.
    This is a wrapper around the existing summarize function.
    """
    summarize(tokenizer, tokens_timelines, logger=logger)


if __name__ == "__main__":
    """
    Example usage of the pure tokenizer with CLIF data processor.
    This demonstrates how to use the new architecture.
    """
    import pathlib
    import sys
    sys.path.append('..')
    from preprocessing.clif import CLIFDataProcessor
    
    # Configuration
    data_dir = "/gpfs/data/bbj-lab/users/burkh4rt/development-sample/raw"
    config_file = "../config/config-20.yaml"
    
    # Create output directory
    # output_dir = pathlib.Path(data_dir).parent / "tokenized-timelines-pure"
    # output_dir.mkdir(exist_ok=True)
    
    # Create CLIF data processor
    clif_processor = CLIFDataProcessor(
        data_dir=data_dir,
        config_file=config_file
    )
    
    # Create pure tokenizer
    tokenizer = PureTokenizer(
        max_padded_len=1024,
        quantizer="deciles",
        include_time_spacing_tokens=False,
        cut_at_24h=False,
        day_stay_filter=True,
    )
    
    # Set the tokenizer reference in the processor (for vocabulary access)
    clif_processor.tokenizer = tokenizer
    
    # Generate tokenized timelines
    print("Generating tokenized timelines using pure tokenizer...")
    tt = tokenizer.get_tokens_timelines(clif_processor)
    
    # Create summary
    create_timeline_summary(tokenizer, tt)
    
    # Save results
    # tt.write_parquet(output_dir / "tokens_timelines_pure.parquet")
    # tokenizer.vocab.save(output_dir / "vocab_pure.gzip")
    # print(f"Pure tokenizer results saved to: {output_dir}")
    
    # Compare with old tokenizer for validation
    print("\n" + "="*50)
    print("COMPARISON WITH OLD TOKENIZER")
    print("="*50)
    
    from fms_ehrs.framework.tokenizer import ClifTokenizer
    from fms_ehrs.framework.tokenizer0 import summarize
    
    # Old tokenizer for comparison
    tkzr_old = ClifTokenizer(
        day_stay_filter=True, 
        data_dir=data_dir
    )
    tt_old = tkzr_old.get_tokens_timelines()
    summarize(tkzr_old, tt_old)
    
    # Save old results for comparison
    # tt_old.write_parquet(output_dir / "tokens_timelines_old.parquet")
    # tkzr_old.vocab.save(output_dir / "vocab_old.gzip")
    
    print(f"\nBoth tokenizers completed successfully!")
    # print(f"Results saved to: {output_dir}")
    print(f"- tokens_timelines_pure.parquet (new architecture)")
    print(f"- tokens_timelines_old.parquet (old architecture)")
    print(f"- vocab_pure.gzip (new vocabulary)")
    print(f"- vocab_old.gzip (old vocabulary)")
