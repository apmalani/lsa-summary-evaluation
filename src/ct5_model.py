import evaluate
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_scheduler, AutoModelForSeq2SeqLM
from datasets import load_from_disk
from lsa_evaluate import Evaluater
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download
from torch.cuda.amp import autocast
from huggingface_hub import HfApi
import nltk
import logging

model_checkpoint = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_checkpoint, legacy = False)

tokenized_datasets = load_from_disk('src/tokens')

class CustomT5Model(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.evaluater = Evaluater()
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name, *model_args, **kwargs):
        model = super(CustomT5Model, cls).from_pretrained(pretrained_model_name, *model_args, **kwargs)
        model.evaluater = Evaluater()
        return model
    
    def compute_lsa_score(self, references, summaries):
        if len(references) != len(summaries):
            raise ValueError('batches offset')
        
        scores = {
            'main_topic': 0,
            'term_sig': 0
        }
        for reference, summary in zip(references, summaries):
            self.evaluater.set_ref_sum(reference, summary)
            scores['main_topic'] += self.evaluater.execute_main_topic('synonyms')
            scores['term_sig'] += self.evaluater.execute_term_sig('synonyms')

        for key in scores:
            scores[key] /= len(summaries)

        lsa_loss = 1 - (scores['main_topic'] + scores['term_sig']) / 2
        return torch.tensor(lsa_loss, dtype=torch.float32, requires_grad=True)

    def compute(self, input_ids, attention_mask = None, labels = None):
        generated_outputs = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=150,
            min_length=40,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2
        )

        generated_summaries = tokenizer.batch_decode(generated_outputs, skip_special_tokens = True)

        decoded_references = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        loss = self.compute_lsa_score(decoded_references, generated_summaries)

        return loss

    def traditional_compute_lsa_score(self, references, summaries):
        pass

model = CustomT5Model.from_pretrained(model_checkpoint)
batch_size = 8
gradient_accumulation_steps = 2

def collate_fn(features):
    return {k: torch.stack([f[k] for f in features], dim=0) for k in features[0].keys()}

train_dataloader = DataLoader(
    tokenized_datasets['train'].select(range(10)),
    shuffle = True,
    collate_fn = collate_fn,
    batch_size = batch_size
)

eval_dataloader = DataLoader(
    tokenized_datasets['test'].select(range(2)),
    collate_fn = collate_fn,
    batch_size = batch_size
)

optimizer = AdamW(model.parameters(), lr = 3e-4, weight_decay=0.01)

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

num_train_epochs = 10
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer = optimizer,
    num_warmup_steps = 0,
    num_training_steps = num_training_steps
)

progress_bar = tqdm(range(num_training_steps))

api = HfApi()
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
repo_name = 'apmalani/lsa-summary-evaluation'
output_dir = 'results-finetined-lsa-1'

rouge_score = evaluate.load('rouge')

api.create_repo(repo_name, exist_ok = True)

def postprocess_text(preds, labels):
    preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in labels]
    return preds, labels

for epoch in range(num_train_epochs):
    model.train()
    total_loss = 0

    for step, batch in enumerate(train_dataloader):
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}

        attention_mask = batch.get('attention_mask', torch.ones_like(batch['input_ids']))

        with accelerator.autocast():
            loss = model.compute(
                input_ids=batch['input_ids'],
                attention_mask=attention_mask,
            )

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
        attention_mask = batch.get('attention_mask', torch.ones_like(batch['input_ids']))

        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=attention_mask,
                max_length=150,
                min_length=40,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2
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
    result = {k: round(v * 100, 2) for k, v in result.items()}
    print(f"Epoch {epoch+1} Evaluation:", result)

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        api.upload_folder(
            folder_path = output_dir,
            repo_id = repo_name,
            commit_message = f'Training in progress at epoch {epoch+1}'
        )