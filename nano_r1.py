import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel
from accelerate import Accelerator
from vllm import LLM, SamplingParams
from datetime import datetime

class GRPODataset(Dataset):
    """Dummy dataset for GRPO training"""
    def __init__(self, prompts):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx]}

class GRPOTrainer:
    def __init__(
        self, 
        model: torch.nn.Module, 
        train_dataset: Dataset,
        reward_fn,  # Callable: takes (prompts, outputs) -> rewards
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
        self.reward_fn = reward_fn
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

    def train(self, num_epochs=3):
        dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.model.train()

        for epoch in range(num_epochs):
            for batch in dataloader:
                prompts = batch["prompt"]
                old_outputs = self.generate(prompts)
                rewards = torch.tensor(self.reward_fn(prompts, old_outputs), dtype=torch.float32).to(self.device)

                inputs = self.model.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
                new_outputs = self.model(**inputs, labels=inputs["input_ids"])
                new_log_probs = F.log_softmax(new_outputs.logits, dim=-1).gather(2, inputs["input_ids"].unsqueeze(2)).squeeze()

                loss = self.compute_grpo_loss(new_log_probs, new_log_probs.detach(), rewards)
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()

                print(f"Epoch {epoch}, Loss: {loss.item()}")

# Example usage
if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    train_data = GRPODataset(["What is the capital of France?", "Solve: 3 + 5"])
    
    def reward_function(prompts, outputs):
        """Simple dummy reward function"""
        return [1.0 if "Paris" in o or "8" in o else 0.0 for o in outputs]

    trainer = GRPOTrainer(model, train_data, reward_function, batch_size=2, use_vllm=False)
    trainer.train(num_epochs=1)
