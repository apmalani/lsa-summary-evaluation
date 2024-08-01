from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, get_scheduler, AutoConfig
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
import huggingface_hub
import torch
import numpy as np
from tqdm.auto import tqdm
import main
import evaluate
import nltk

rouge_score = evaluate.load('rouge')

cnn_dailymail = load_dataset('abisee/cnn_dailymail', '3.0.0')

model_checkpoint = "google-t5/t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

max_input_length = 2048
max_label_length = 30

# def preprocess(dataset):
#     model_inputs = tokenizer(
#         dataset['article'],
#         max_length = max_input_length,
#         truncation = True,
#         padding = 'max_length'
#     )

#     labels = tokenizer(
#         dataset['highlights'],
#         max_length = max_label_length,
#         truncation = True,
#         padding = 'max_length'
#     )

#     model_inputs['labels'] = labels['input_ids']
#     return model_inputs

# tokenized_datasets = cnn_dailymail.map(preprocess, batched = True, remove_columns=cnn_dailymail['train'].column_names)

# tokenized_datasets.set_format("torch")

# tokenized_datasets.save_to_disk('tokens')

tokenized_datasets = load_from_disk('tokens')

class CustomSeq2SeqModel(AutoModelForSeq2SeqLM):
    def __init__(self, config):
        super().__init__(config)
        self.evaluater = main.Evaluater()

    def compute_lsa_loss(self, summaries, references):
        scores = {'main topic': 0, 'term sig': 0}
        num_samples = len(summaries)

        for ref, sum in zip(references, summaries):
            self.evaluater.set_ref_sum(ref, sum)
            scores['main topic'] += self.evaluater.execute_main_topic("synonyms")
            scores['term sig'] += self.evaluater.execute_term_sig("synonyms")

        for key in scores:
            scores[key] /= num_samples

        lsa_loss = 1 - (scores['main topic'] + scores['term sig']) / 2

        return torch.tensor(lsa_loss, dtype=torch.float32)
    
    def forward(self, input_ids = None, attention_mask = None, labels = None):
        outputs = super().forward(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
        logits = outputs.logits

        predictions = torch.argmax(logits, dim = -1)
        print(predictions)
        summaries = tokenizer.batch_decode(predictions, skip_special_tokens = True)
        references = tokenizer.batch_decode(labels, skip_special_tokens = True)

        lsa_loss = self.compute_lsa_loss(summaries, references)

        outputs.loss = lsa_loss
        return outputs

model = CustomSeq2SeqModel.from_pretrained(model_checkpoint)

batch_size = 4

# data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

def custom_collate_fn(features):
    batch = {}
    for key in features[0].keys():
        if isinstance(features[0][key], torch.Tensor):
            batch[key] = torch.stack([f[key] for f in features])
        elif isinstance(features[0][key], list):
            batch[key] = torch.tensor([f[key] for f in features], dtype=torch.int64)
        else:
            batch[key] = torch.tensor([f[key] for f in features])
    return batch

train_dataloader = DataLoader(
    tokenized_datasets["test"],
    shuffle = True,
    collate_fn = custom_collate_fn,
    batch_size = batch_size
)

eval_dataloader = DataLoader(
    tokenized_datasets["validation"],
    collate_fn = custom_collate_fn,
    batch_size = batch_size
)

optimizer = AdamW(model.parameters(), lr = 2e-5)

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

num_train_epochs = 12
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer = optimizer,
    num_warmup_steps = 0,
    num_training_steps = num_training_steps
)

progress_bar = tqdm(range(num_training_steps))

repo_name = 'apmalani/lsa-summary-evaluation'
output_dir = 'results-finetined-lsa-1'
repo = huggingface_hub.Repository(output_dir, clone_from = repo_name)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # ROUGE expects a newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


for epoch in range(num_train_epochs):
    model.train()

    print("Beg. of Epic ", epoch)

    for step, batch in enumerate(train_dataloader):
        print("reaches here")
        outputs = model(**batch)
        print("1")
        loss = outputs.loss
        print("2")
        accelerator.backward(loss)
        print("3")
        # TODO: ISSUE IN LOSS FUNCTION
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        print("also here")
    
    model.eval()

    print('reached eval')

    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask = batch["attention_mask"]
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim = 1, pad_index = tokenizer.pad_token_id
            )
            labels = batch["id"]

            labels = accelerator.pad_across_processes(
                batch["id"], dim = 1, pad_index = tokenizer.pad_token_id
            )

            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
            labels = accelerator.gather(labels).cpu().numpy()

            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens = True
            )
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens = True)

            reference_texts = batch["article"]

            decoded_preds, decoded_labels = postprocess_text(
                decoded_preds, decoded_labels
            )

            # add batch of results to be computed
            rouge_score.add_batch(predictions = decoded_preds, references = decoded_labels)


    # compute results
    result = rouge_score.compute()
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    result = {k: round(v, 4) for k, v in result.items()}
    print(f"Epoch {epoch}:", result)

    # extract the median results

    # print epoch and result

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function = accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        repo.push_to_hub(
            commit_message = f'Training in progress at epoch {epoch}', blocking = False
        )