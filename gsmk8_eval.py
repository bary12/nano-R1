from pathlib import Path
import pandas as pd

strip = lambda answer: answer[answer.rfind("#### ") + 5:].strip()
EVAL_FN = lambda model_answer, truth: model_answer.strip() == truth

splits = {'train': 'main/train-00000-of-00001.parquet', 'test': 'main/test-00000-of-00001.parquet'}

trainset = pd.read_parquet("hf://datasets/openai/gsm8k/" + splits["train"])
testset = pd.read_parquet("hf://datasets/openai/gsm8k/" + splits["test"])

train_points = []
test_points = []
for dataset, datapoints in zip((trainset, testset), (train_points, test_points)):
    for _, row in dataset.iterrows():
        datapoints.append(
            {
                "prompt": row["question"],
                "answer": strip(row["answer"]),
                "eval_fn": EVAL_FN
            }
        )