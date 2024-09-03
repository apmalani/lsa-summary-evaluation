import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from lsa_evaluate import Evaluater
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from rouge_score import rouge_scorer
from huggingface_hub import HfApi
import logging

class SummaryDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]['article']
        highlights = self.dataset[idx]['highlights']
        return text, highlights

class SummaryEnvironment:
    def __init__(self, text):
        self.text = text
        self.evaluator = Evaluater()

    def get_reward(self, summary):
        self.evaluator.set_ref_sum(self.text, summary)
        lsa_score = self.evaluator.execute_main_topic()
        return lsa_score

class SummaryAgent:
    def __init__(self, model_name):
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)

    def generate_summary(self, text):
        inputs = self.tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = self.model.generate(inputs.input_ids, max_length=150, min_length=40, num_beams=4, early_stopping=True, no_repeat_ngram_size=2)
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def train_step(self, text, optimizer):
        self.model.train()
        inputs = self.tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)

        with torch.no_grad():
            output_ids = self.model.generate(inputs.input_ids, max_length=150, min_length=40, num_beams=4, early_stopping=True, no_repeat_ngram_size=2)
        
        generated_summary = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        env = SummaryEnvironment(text)
        reward = env.get_reward(generated_summary)

        reward_tensor = torch.tensor(reward, requires_grad = True)

        loss =  1 - reward_tensor
 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        return loss.item(), reward

def evaluate_model(agent, dataloader):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

    for batch in dataloader:
        for i in range(max(1, len(batch) // 10)):
            summary = agent.generate_summary(batch[0][i])
            scores = scorer.score(batch[1][i], summary)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)

    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)

    return avg_rouge1, avg_rouge2, avg_rougeL

dataset = load_dataset("cnn_dailymail", "3.0.0", split='train')
summary_dataset = SummaryDataset(dataset.select(range(10)))
dataloader = DataLoader(summary_dataset, batch_size=8, shuffle=True)

agent = SummaryAgent("apmalani/lsa-summary-evaluation")
optimizer = torch.optim.Adam(agent.model.parameters(), lr=3e-5)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

num_epochs = 3
for epoch in range(num_epochs):
    for idx, batch in enumerate(dataloader):
        for i in range(len(batch[0])):
            loss, reward = agent.train_step(batch[0][i], optimizer)
        print(f"Epoch {epoch + 1}, Batch {idx + 1}, Reward: {reward}")

    avg_rouge1, avg_rouge2, avg_rougeL = evaluate_model(agent, dataloader)
    print(f"Evaluating Epoch {epoch + 1}, ROUGE-1: {avg_rouge1}, ROUGE-2: {avg_rouge2}, ROUGE-L: {avg_rougeL}")

    agent.model.save_pretrained("results-finetuned-lsa")
    agent.tokenizer.save_pretrained("results-finetuned-lsa")

    api = HfApi()
    api.upload_folder(
        folder_path = 'results-finetuned-lsa',
        repo_id = 'apmalani/lsa-summary-evaluation',
        repo_type = 'model',
        commit_message = f'Training in progress at epoch {epoch+1}'
    )