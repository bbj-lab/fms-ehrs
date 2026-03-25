To load data from a given split and extract prefix and suffix tokens:

```py
import pathlib
import polars as pl
from fms_ehrs.framework.vocabulary import Vocabulary

splits = ("train", "val", "test")
data_dir = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/data-mimic/W++-tokenized")
vocab = Vocabulary().load(data_dir.joinpath("train", "vocab.gzip"))
dfs = dict()
for s in splits:
    dfs[s] = pl.read_parquet(
        data_dir.joinpath(s, "tokens_timelines.parquet")
    ).with_columns(
        prefix=pl.col("tokens")
        .list.head(n=6)
        .list.eval(
            pl.element().map_elements(
                lambda v: vocab.reverse[v], return_dtype=pl.String
            )
        ),
        suffix=pl.col("tokens")
        .list.tail(2)
        .list.eval(
            pl.element().map_elements(
                lambda v: vocab.reverse[v], return_dtype=pl.String
            )
        ),
    )
```
