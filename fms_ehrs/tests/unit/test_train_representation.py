import torch

from fms_ehrs.scripts.train_representation import RepresentationDataCollator


def test_representation_data_collator_masks_pad_labels():
    collator = RepresentationDataCollator(pad_token_id=0)
    batch = collator(
        [
            {"input_ids": torch.tensor([5, 6, 0, 0], dtype=torch.long)},
            {"input_ids": torch.tensor([7, 8, 9, 0], dtype=torch.long)},
        ]
    )

    assert batch["input_ids"].tolist() == [[5, 6, 0, 0], [7, 8, 9, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 0, 0], [1, 1, 1, 0]]
    assert batch["labels"].tolist() == [[5, 6, -100, -100], [7, 8, 9, -100]]
