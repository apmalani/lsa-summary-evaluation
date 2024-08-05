from transformers import AutoTokenizer, MT5ForConditionalGeneration, get_scheduler
from transformers.modeling_outputs import Seq2SeqLMOutput
from datasets import load_from_disk
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
from torch.amp import autocast

# Load ROUGE metric
rouge_score = evaluate.load('rouge')

model_checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Load tokenized datasets
tokenized_datasets = load_from_disk('tokens')

class CustomSeq2SeqModel(MT5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.evaluater = main.Evaluater()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super(CustomSeq2SeqModel, cls).from_pretrained(*args, **kwargs)
        model.evaluater = main.Evaluater()
        return model

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
        return torch.tensor(lsa_loss, dtype=torch.float32, requires_grad=True)
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        
        if labels is not None:
            predictions = torch.argmax(outputs.logits, dim=-1)
            summaries = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            references = tokenizer.batch_decode(labels, skip_special_tokens=True)
            lsa_loss = self.compute_lsa_loss(summaries, references)
            outputs.loss = outputs.loss + lsa_loss

        return outputs

model = CustomSeq2SeqModel.from_pretrained(model_checkpoint)

batch_size = 2  # Increased batch size
gradient_accumulation_steps = 2  # Added gradient accumulation

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def collate_fn(features):
    return {k: torch.stack([f[k].clone().detach() for f in features]) for k in features[0].keys()}

train_dataloader = DataLoader(
    tokenized_datasets["test"].select(range(10)),  # Increased dataset size
    shuffle=True,
    collate_fn=collate_fn,
    batch_size=batch_size
)

eval_dataloader = DataLoader(
    tokenized_datasets["validation"].select(range(2)),  # Increased evaluation dataset size
    collate_fn=collate_fn,
    batch_size=batch_size
)

optimizer = AdamW(model.parameters(), lr=2e-5)

accelerator = Accelerator(mixed_precision='fp16' if torch.backends.mps.is_available() else 'no')
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

num_train_epochs = 8
num_update_steps_per_epoch = len(train_dataloader) // gradient_accumulation_steps
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

progress_bar = tqdm(range(num_training_steps))

repo_name = 'apmalani/lsa-summary-evaluation'
output_dir = 'results-finetined-lsa-1'
repo = huggingface_hub.Repository(output_dir, clone_from=repo_name)

def postprocess_text(preds, labels):
    preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in labels]
    return preds, labels

for epoch in range(num_train_epochs):
    model.train()
    total_loss = 0

    for step, batch in enumerate(train_dataloader):
        with autocast(device_type=device.type):
            outputs = model(**batch)
            loss = outputs.loss / gradient_accumulation_steps
        
        accelerator.backward(loss)
        total_loss += loss.item()

        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            progress_bar.set_description(f"Loss: {total_loss:.4f}")
            total_loss = 0

    model.eval()
    for batch in eval_dataloader:
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=300
            )

        labels = batch["labels"]
        generated_tokens = accelerator.pad_across_processes(generated_tokens, dim=1, pad_index=tokenizer.pad_token_id)
        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=tokenizer.pad_token_id)

        generated_tokens, labels = accelerator.gather_for_metrics((generated_tokens, labels))
        generated_tokens = generated_tokens.cpu().numpy()
        labels = labels.cpu().numpy()

        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        rouge_score.add_batch(predictions=decoded_preds, references=decoded_labels)

    result = rouge_score.compute()
    result = {k: round(v, 4) for k, v in result.items()}
    print(f"Epoch {epoch}:", result)

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)

    for param in unwrapped_model.parameters():
        if not param.is_contiguous():
            param.data = param.contiguous()

    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        repo.push_to_hub(commit_message=f'Training in progress at epoch {epoch}', blocking=False)
