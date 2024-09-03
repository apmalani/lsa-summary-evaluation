from transformers import T5Tokenizer
from datasets import load_dataset

model_checkpoint = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_checkpoint, legacy = False)

max_input_length = 512
max_label_length = 128

cnn_dailymail = load_dataset('abisee/cnn_dailymail', '3.0.0')


def preprocess(dataset):
    inputs = ["summarize: " + article for article in dataset['article']]

    model_inputs = tokenizer(
        inputs,
        max_length = max_input_length,
        truncation = True,
        return_attention_mask = True,
        padding = True
    )

    labels = tokenizer(
        dataset['highlights'],
        max_length = max_label_length,
        truncation = True,
        return_attention_mask = True,
        padding = True
    )

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_datasets = cnn_dailymail.map(preprocess, batched = True, remove_columns=cnn_dailymail['train'].column_names)

def filter_long_sequences(example):
    return len(example['input_ids']) <= max_input_length and len(example['labels']) <= max_label_length

tokenized_datasets = tokenized_datasets.filter(filter_long_sequences)

tokenized_datasets.set_format("torch", columns=['input_ids', 'labels'])

tokenized_datasets.save_to_disk('src/tokens')