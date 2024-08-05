from transformers import AutoTokenizer
from datasets import load_dataset

model_checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

max_input_length = 2048
max_label_length = 2048

cnn_dailymail = load_dataset('abisee/cnn_dailymail', '3.0.0')

def preprocess(dataset):
    model_inputs = tokenizer(
        dataset['article'],
        max_length = max_input_length,
        truncation = True,
        padding = True
    )

    labels = tokenizer(
        dataset['highlights'],
        max_length = max_label_length,
        truncation = True,
        padding = True
    )

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_datasets = cnn_dailymail.map(preprocess, batched = True, remove_columns=cnn_dailymail['train'].column_names)

def filter_long_sequences(example):
    return len(example['input_ids']) <= max_input_length and len(example['labels']) <= max_label_length

tokenized_datasets = tokenized_datasets.filter(filter_long_sequences)

tokenized_datasets.set_format("torch")

tokenized_datasets.save_to_disk('tokens')