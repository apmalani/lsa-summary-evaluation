from transformers import AutoTokenizer, mT5ForConditionalGeneration, mT5Config, get_scheduler, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import Seq2SeqLMOutput
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

model_checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

tokenized_datasets = load_from_disk('tokens')

class CustomSeq2SeqModel(mT5ForConditionalGeneration):
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

        print(lsa_loss)

        return torch.tensor(lsa_loss, dtype = torch.float32, requires_grad = True)
    
    def forward(self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        predictions = torch.argmax(logits, dim = -1)        
        summaries = tokenizer.batch_decode(predictions, skip_special_tokens = True)
        print(summaries)
        references = tokenizer.batch_decode(labels, skip_special_tokens = True) if labels is not None else None
        print(references)

        if labels is not None:
            lsa_loss = self.compute_lsa_loss(summaries, references)
        else:
            lsa_loss = None

        return Seq2SeqLMOutput(
            loss=lsa_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
    
model = CustomSeq2SeqModel.from_pretrained(model_checkpoint)

batch_size = 2

def collate_fn(features):
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
    tokenized_datasets["test"].select(range(10)),
    shuffle = True,
    collate_fn = collate_fn,
    batch_size = batch_size
)

eval_dataloader = DataLoader(
    tokenized_datasets["validation"].select(range(2)),
    collate_fn = collate_fn,
    batch_size = batch_size
)

optimizer = AdamW(model.parameters(), lr = 2e-5)

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

num_train_epochs = 8
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

    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def save_model_contiguous(model, output_dir):
    for param in model.parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()
    model.save_pretrained(output_dir)

for epoch in range(num_train_epochs):
    model.train()

    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    
    model.eval()

    for step, batch in enumerate(eval_dataloader):
        print(batch)
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask = batch["attention_mask"],
                max_new_tokens = 300
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim = 1, pad_index = tokenizer.pad_token_id
            )

            labels = batch["old_summaries"]

            labels = accelerator.pad_across_processes(
                batch["old_summaries"], dim = 1, pad_index = tokenizer.pad_token_id
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

            decoded_preds, decoded_labels = postprocess_text(
                decoded_preds, decoded_labels
            )

            rouge_score.add_batch(predictions = decoded_preds, references = decoded_labels)

    result = rouge_score.compute()
    # result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    result = {k: round(v, 4) for k, v in result.items()}
    print(f"Epoch {epoch}:", result)

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    save_model_contiguous(unwrapped_model, output_dir)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        repo.push_to_hub(
            commit_message = f'Training in progress at epoch {epoch}', blocking = False
        )