# `run_deepseek_v1.py`
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(30)

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")

chat = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
    {"role": "user", "content": "I'd like to show off how chat templating works!"},
]


model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-V3", dtype=torch.bfloat16
)
print(f"Original Model: {model}")
# inputs = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
# import time
# start = time.time()
# outputs = model.generate(inputs, max_new_tokens=50)
# print(tokenizer.batch_decode(outputs))
# print(time.time()-start)
