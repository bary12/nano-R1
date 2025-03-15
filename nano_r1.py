import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from vllm import LLM, SamplingParams
import re

class GRPODataset(Dataset):
    """Dataset for GRPO training with questions and answers"""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"question": self.data[idx]["question"], "answer": self.data[idx]["answer"]}

class GRPOTrainer:
    def __init__(
        self, 
        model: torch.nn.Module, 
        train_dataset: Dataset,
        batch_size: int = 8, 
        num_generations: int = 4,
        lr: float = 5e-6,
        epsilon: float = 0.2,
        beta: float = 0.01,  # KL penalty
        use_vllm: bool = False,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.batch_size = batch_size
        self.num_generations = num_generations
        self.lr = lr
        self.epsilon = epsilon
        self.beta = beta
        self.use_vllm = use_vllm
        self.device = device

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.accelerator = Accelerator()
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

        if self.use_vllm:
            self.llm = LLM(model=self.model, dtype="float16", max_model_len=2048)

    def generate(self, prompts):
        """Generate outputs using vLLM or standard model inference"""
        if self.use_vllm:
            sampling_params = SamplingParams(max_tokens=128, n=self.num_generations, temperature=0.7, top_p=0.9)
            return [output.outputs for output in self.llm.generate(prompts, sampling_params)]
        else:
            inputs = self.model.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=256, num_return_sequences=self.num_generations)
            return [self.model.tokenizer.decode(out, skip_special_tokens=True) for out in outputs]

    def compute_grpo_loss(self, old_log_probs, new_log_probs, rewards):
        """Compute GRPO loss based on DeepSeek R1 formula"""
        advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        ratio = (new_log_probs - old_log_probs).exp()
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()

        kl_div = F.kl_div(new_log_probs, old_log_probs, reduction="batchmean", log_target=True)
        return loss + self.beta * kl_div

    def reward_function(self, prompts, outputs, answers):
        """Reward function with format and correctness evaluation"""
        rewards = []
        for output, answer in zip(outputs, answers):
            format_reward = 1.0 if re.match(r"<think>.*</think>\n<answer>.*</answer>", output, re.DOTALL) else 0.0
            extracted_answer = re.search(r"<answer>(.*?)</answer>", output)
            correctness_reward = 1.0 if extracted_answer and extracted_answer.group(1).strip() == answer.strip() else 0.0
            rewards.append(format_reward + correctness_reward)
        return rewards

    def train(self, num_epochs=3):
        dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.model.train()

        for epoch in range(num_epochs):
            for batch in dataloader:
                prompts = batch["question"]
                answers = batch["answer"]
                old_outputs = self.generate(prompts)
                rewards = torch.tensor(self.reward_function(prompts, old_outputs, answers), dtype=torch.float32).to(self.device)

                inputs = self.model.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
                new_outputs = self.model(**inputs, labels=inputs["input_ids"])
                new_log_probs = F.log_softmax(new_outputs.logits, dim=-1).gather(2, inputs["input_ids"].unsqueeze(2)).squeeze()

                loss = self.compute_grpo_loss(new_log_probs, new_log_probs.detach(), rewards)
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()

                print(f"Epoch {epoch}, Loss: {loss.item()}")

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    train_data = GRPODataset([
        {"question": "What is the capital of France?", "answer": "Paris"},
        {"question": "Solve: 3 + 5", "answer": "8"}
    ])

    trainer = GRPOTrainer(model, train_data, batch_size=2, use_vllm=False)
    trainer.train(num_epochs=1)
