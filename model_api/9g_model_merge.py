from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch, os, shutil, json

base = "/share_data/data1/qy/model/9g/9g_8b_thinking"
adapter = "/share_data/data1/qy/sft_model/9g/9g_8b_thinking/OCNLILoRA/20251001225743/checkpoint-500"
out = "/share_data/data1/qy/sft_model/9g/9g_8b_thinking/9g_8b_thinking_lora_v2"

os.makedirs(out, exist_ok=True)

tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
tok.save_pretrained(out)

# 在 CPU 上加载，避免显存占用
model = AutoModelForCausalLM.from_pretrained(
    base,
    torch_dtype=torch.bfloat16,   # 或 torch.float16/torch.float32
    trust_remote_code=True,
    attn_implementation="eager",  # 或 "flash_attention_2"，CPU 下用 eager 更稳妥
    device_map=None  # 关键：不做自动映射到 GPU
)
model.to("cpu")

# 加载 LoRA 适配器并在 CPU 上合并
model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()

# 保存合并后的全量权重
model.save_pretrained(out, safe_serialization=True)

# 复制额外文件
def copy_if_exists(fname):
    src = os.path.join(adapter, fname)
    if os.path.exists(src):
        shutil.copy(src, out)

for fname in ["chat_template.jinja", "added_tokens.json", "special_tokens_map.json", "tokenizer_config.json", "generation_config.json"]:
    copy_if_exists(fname)

# 调整 config 里的注意力实现为推理友好
cfg_path = os.path.join(out, "config.json")
if os.path.exists(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    cfg["_attn_implementation"] = "flash_attention_2"  # GPU 推理时可改回 flash_attn_2
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

print("Merged model saved to:", out)