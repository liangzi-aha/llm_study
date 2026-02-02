# 核心：关闭M5的内存分配器坑，这是解决Invalid buffer size的关键
import os
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 只导入必需的库，不用accelerate/量化相关
from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch

# 模型名
model_name = "Qwen/Qwen2.5-7B-Instruct"

# 加载模型：极简配置，适配M5 24G，无任何复杂参数
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,  # 仅保留Qwen必需的参数
    torch_dtype=torch.bfloat16,  # 低精度，省内存
    device_map='cpu',  # 强制用CPU+内存跑，避开MPS的所有坑
    low_cpu_mem_usage=True  # 省内存关键，24G完全够
)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 极简对话，中文提问（更直观）
messages = [{"role": "user", "content": "你好，简单介绍下自己"}]
# 构造输入
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([input_text], return_tensors='pt')
# 生成回复（极简参数，快出结果）
outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
# 解码输出
response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
# 打印结果
print('模型回复：', response)