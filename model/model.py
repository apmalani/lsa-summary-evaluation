from transformers import AutoTokenizer
from datasets import load_dataset

cnn_dailymail = load_dataset('abisee/cnn_dailymail', '3.0.0')

print(cnn_dailymail)

model_checkpoint = "google-t5/t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

max_input_length = 2048
max_target_length = 30

# def preprocess(articles):
#     model_inputs = tokenizer(
#         articles["article"]
#     )