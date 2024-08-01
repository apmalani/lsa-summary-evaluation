from transformers import AutoTokenizer
from datasets import load_dataset

model_checkpoint = "google/t5-v1_1-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

max_input_length = 2048
max_label_length = 30

cnn_dailymail = load_dataset('abisee/cnn_dailymail', '3.0.0')

def preprocess(dataset):
    model_inputs = tokenizer(
        dataset['article'],
        max_length = max_input_length,
        truncation = True,
        padding = 'max_length'
    )

    labels = tokenizer(
        dataset['highlights'],
        max_length = max_label_length,
        truncation = True,
        padding = 'max_length'
    )

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_datasets = cnn_dailymail.map(preprocess, batched = True, remove_columns=cnn_dailymail['train'].column_names)

tokenized_datasets.set_format("torch")

tokenized_datasets.save_to_disk('tokens')