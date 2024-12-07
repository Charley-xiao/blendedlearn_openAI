import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import load_dataset
import random
import yaml
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np

def load_config(config_file):
    with open(config_file, 'r') as file:
        config_dict = yaml.safe_load(file)
    return config_dict

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class RewardModel:
    def __init__(self, model_names: Dict[str, str], device: str):
        self.device = torch.device("cuda" if device == 'cuda' and torch.cuda.is_available() else "cpu")
        self.models = {}
        self.tokenizers = {}
        for aspect, model_name in model_names.items():
            self.tokenizers[aspect] = AutoTokenizer.from_pretrained(model_name)
            self.models[aspect] = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)

    def compute_reward(self, responses: List[str]):
        rewards = []
        for response in responses:
            aspect_scores = {}
            for aspect, model in self.models.items():
                tokenizer = self.tokenizers[aspect]
                inputs = tokenizer(response, return_tensors="pt", truncation=True, max_length=512).to(self.device)
                outputs = model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
                label_score = scores[:, 1] if model.config.id2label[1] in ["TOXIC", "OFFENSIVE"] else scores[:, 0]
                score = label_score.item()
                aspect_scores[aspect] = score
            reward = self._combine_aspect_scores(aspect_scores)
            rewards.append(torch.tensor(reward))
        return rewards

    def _get_continuous_score(self, score: float, label: str) -> float:
        if label in ["TOXIC", "OFFENSIVE"]:
            return score  # Higher score indicates higher toxicity
        else:
            return 1.0 - score  # Lower toxicity

    def _combine_aspect_scores(self, aspect_scores: Dict[str, float]) -> float:
        weights = {
            'toxicity': 0.5,
            'hate_speech': 0.3,
            'profanity': 0.2,
        }
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        reward = 0.0
        for aspect, score in aspect_scores.items():
            weight = weights.get(aspect, 0.0)
            reward += weight * (1.0 - score)
        return reward

class ConstitutionalAI:
    def __init__(self, ethical_guidelines: List[str]):
        self.guidelines = ethical_guidelines

    def evaluate(self, response: str) -> bool:
        violations = []
        for rule in self.guidelines:
            if self._detect_violation(response, rule):
                violations.append(rule)
        return len(violations) == 0

    def _detect_violation(self, text: str, rule: str) -> bool:
        # Implement advanced detection logic here
        # For example, use regex patterns or NLP models to detect violations
        # Placeholder implementation
        return rule.lower() in text.lower()

class SafeLanguageModelTrainer:
    def __init__(self, config):
        set_seed(config['seed'])
        self.config = config
        self.device = config['device']
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.base_model = AutoModelForCausalLMWithValueHead.from_pretrained(config['model_name']).to(self.device)
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(config['model_name']).to(self.device)
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        # self.model.generation_config = self.model.config
        self.base_model.config.eos_token_id = self.tokenizer.eos_token_id
        self.reward_model = RewardModel(config['toxicity_model_names'], self.device)
        self.ppo_config = PPOConfig(
            learning_rate=float(config['learning_rate']),
            batch_size=int(config['batch_size']),
            mini_batch_size=int(config['mini_batch_size'])
        )
        self.ppo_trainer = PPOTrainer(self.ppo_config, self.base_model, self.model, self.tokenizer)
        self.constitutional_ai = ConstitutionalAI(ethical_guidelines=config['ethical_guidelines'])
        self.curriculum = self._create_curriculum()
        self.current_difficulty = 0

    def _create_curriculum(self):
        # Load datasets with increasing levels of difficulty
        curriculum = []
        # Stage 1: Non-toxic data
        dataset_stage1 = load_dataset('daily_dialog', split='train[:1%]')
        curriculum.append({'dataset': dataset_stage1, 'difficulty': 1})
        # Stage 2: Mixed data
        dataset_stage2 = load_dataset('reddit_tifu', 'short', split='train[:1%]')
        curriculum.append({'dataset': dataset_stage2, 'difficulty': 2})
        # Stage 3: Data likely to contain toxicity
        dataset_stage3 = load_dataset('civil_comments', split='train[:1%]')
        curriculum.append({'dataset': dataset_stage3, 'difficulty': 3})
        return curriculum

    def train(self):
        num_epochs = self.config['num_epochs']
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            self.current_difficulty = min(self.current_difficulty + 1, len(self.curriculum) - 1)
            dataset = self.curriculum[self.current_difficulty]['dataset']
            prompts = self._prepare_prompts(dataset)
            for batch in self._get_batches(prompts, self.config['batch_size']):
                self._train_batch(batch)

    def _prepare_prompts(self, dataset):
        prompts = []
        for data in dataset:
            prompt = data.get('text') or data.get('content') or data.get('documents') or data.get('dialog') or ''
            if prompt:
                prompts.append(prompt.strip())
        return prompts

    def _get_batches(self, data, batch_size):
        random.shuffle(data)
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    def _train_batch(self, prompts):
        # Encode prompts and include attention mask
        encoded_prompts = [self.tokenizer(prompt, return_tensors='pt', max_length=self.config['max_length'], 
                                          truncation=True, padding=True) for prompt in prompts]
        
        prompt_tensors = [encoding['input_ids'].to(self.device) for encoding in encoded_prompts]
        attention_masks = [encoding['attention_mask'].to(self.device) for encoding in encoded_prompts]
    
        response_tensors = []
        for prompt_tensor, attention_mask in zip(prompt_tensors, attention_masks):
            # Remove batch dimension if needed
            prompt_tensor = prompt_tensor.squeeze(0)
            print(prompt_tensor.size(0))
            max_length = 1024
            if prompt_tensor.size(0) > max_length:
                response_tensors.append(prompt_tensor)
                print('Skipped')
                continue
            
            # Pass attention mask to generate
            response = self.ppo_trainer.generate(
                query_tensor=prompt_tensor, 
                eos_token_id=self.tokenizer.eos_token_id, 
                pad_token_id=self.tokenizer.eos_token_id,
                attention_mask=attention_mask,
                max_length=1024
            )
            response_tensors.append(response)
            print(response.shape)
    
        # Decode and process responses
        responses = [self.tokenizer.decode(response_tensor.squeeze(), skip_special_tokens=True) for response_tensor in response_tensors]
    
        safe_responses = []
        for response in responses:
            if not self.constitutional_ai.evaluate(response):
                response = self._adjust_response(response)
            safe_responses.append(response)
    
        # Compute rewards and train
        rewards = self.reward_model.compute_reward(safe_responses)
        print(rewards)
        # prompt_tensors = [prompt_tensor.squeeze(0) for prompt_tensor in prompt_tensors]
        # Pad prompt_tensors to have the same length with response_tensors
        # torch.Size([1, 358]) -> torch.Size([1, 1024])
        prompt_tensors = [torch.cat([prompt_tensor, torch.zeros(1, 1024 - prompt_tensor.size(1)).long().to(self.device)], dim=1) for prompt_tensor in prompt_tensors]
        print(prompt_tensors)
        print(prompt_tensors[0].shape)
        print(response_tensors)
        print(response_tensors[0].shape)
        prompt_tensors = [prompt_tensor.squeeze(0) for prompt_tensor in prompt_tensors]
        response_tensors = [response_tensor.squeeze(0) for response_tensor in response_tensors]
        print(prompt_tensors)
        print(response_tensors)
        self.ppo_trainer.step(prompt_tensors, response_tensors, rewards)
    
        # Log responses
        for prompt, response, reward in zip(prompts, safe_responses, rewards):
            # print(f"Prompt: {prompt}")
            # print(f"Response: {response}")
            print(f"Reward: {reward:.4f}\n")


    def _adjust_response(self, response):
        return "[Content removed due to violation of ethical guidelines.]"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='train', help='train or inference')
    args = parser.parse_args()
    config_type = args.type
    config_file = os.path.join('cfg', f'{config_type}.yaml')
    config = load_config(config_file)
    if config_type == 'train':
        trainer = SafeLanguageModelTrainer(config)
        trainer.train()
    elif config_type == 'inference':
        pass
