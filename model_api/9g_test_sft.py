from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base_model = "/share_data/data1/qy/model/9g/9g_8b_thinking"
lora_ckpt = "/share_data/data1/qy/sft_model/9g/9g_8b_thinking/OCNLILoRA/20251001223248/checkpoint-1"

# 建议从 LoRA ckpt 加载 tokenizer，以保持你训练后保存的 special tokens 和 chat template 一致
tokenizer = AutoTokenizer.from_pretrained(lora_ckpt, trust_remote_code=True)

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,     # 你的训练是 bf16，可改为 fp16 或 fp32 根据显存情况
    device_map="auto",              # 自动放到可用 GPU
    trust_remote_code=True
)

# 叠加 LoRA 适配器
model = PeftModel.from_pretrained(model, lora_ckpt)
model.eval()

# 构造输入（如果你有 chat_template，建议使用 apply_chat_template）
prompt = "请用中文解释什么是长上下文注意力，并给一个简单示例。"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.8,
        top_p=0.9
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))

messages = [
    {"role": "system", "content": "你是一个乐于助人的助手。"},
    {"role": "user", "content": "介绍一下LoRA是什么？"},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True  # 在末尾加上assistant起始标记
)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
print(tokenizer.decode(output[0], skip_special_tokens=True))

from transformers import AutoModelForCausalLM, BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(load_in_4bit=True)  # 或 load_in_8bit=True
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(model, lora_ckpt)
model.eval()
