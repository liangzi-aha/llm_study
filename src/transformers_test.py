import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# =====================================================
# 直接指定本地模型路径（使用你已下载的ModelScope缓存）
# =====================================================

# 你的ModelScope下载路径
local_model_path = os.path.expanduser("~/.cache/modelscope/hub/models/Qwen/Qwen2.5-7B-Instruct")

print(f"📂 从本地路径加载模型: {local_model_path}")
print(f"📁 路径存在吗? {os.path.exists(local_model_path)}")

# 检查模型文件
print("\n📋 模型文件列表:")
for file in os.listdir(local_model_path)[:5]:  # 只显示前5个
    print(f"  - {file}")

# 加载分词器（从本地）
print("\n📝 加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(
    local_model_path,  # 直接用本地路径
    trust_remote_code=True
)

# 加载模型（从本地）
print("📥 加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,  # 直接用本地路径
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True
)

print(f"✅ 模型加载完成！设备: {model.device}")

# 测试对话
messages = [{"role": "user", "content": "你好，简单介绍下自己"}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([input_text], return_tensors='pt')

print("⏳ 生成回复中...")
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True
)

response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
print(f"\n🤖 模型回复: {response}")