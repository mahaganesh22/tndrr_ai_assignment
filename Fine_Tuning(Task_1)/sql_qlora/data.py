"""Dataset loading, cleaning, splitting, and chat-template formatting."""

from datasets import Dataset, DatasetDict, load_dataset

from .config import Config, SYSTEM_PROMPT, user_prompt


def load_splits(config: Config) -> DatasetDict:
    raw = load_dataset(config.dataset_name, split="train")
    frame = raw.to_pandas().dropna(subset=["context", "question", "answer"])
    frame = frame.drop_duplicates(
        subset=["context", "question", "answer"]
    ).reset_index(drop=True)

    required = config.train_size + config.validation_size + config.test_size
    if len(frame) < required:
        raise ValueError(
            f"Need at least {required} unique rows; dataset has {len(frame)}."
        )

    dataset = Dataset.from_pandas(frame, preserve_index=False)
    held_out = dataset.train_test_split(
        test_size=config.test_size, seed=config.seed, shuffle=True
    )
    train_validation = held_out["train"].train_test_split(
        test_size=config.validation_size, seed=config.seed, shuffle=True
    )
    train = train_validation["train"].shuffle(seed=config.seed).select(
        range(config.train_size)
    )
    splits = DatasetDict(
        train=train,
        validation=train_validation["test"],
        test=held_out["test"],
    )
    return splits


def format_splits(splits: DatasetDict, tokenizer) -> DatasetDict:
    def format_row(row):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": user_prompt(row["context"], row["question"]),
            },
            {"role": "assistant", "content": str(row["answer"])},
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    return DatasetDict(
        {
            name: split.map(format_row, remove_columns=split.column_names)
            for name, split in splits.items()
        }
    )
