import json

# 假设文件叫 data.json
with open("/home/hubing/SimPO/datasets/gemma2_ultrafeedback/inpo_iter1/all_outputs_rm.json", "r", encoding="utf-8") as f:
    data = json.load(f)

empty_cases = []

for i, sample in enumerate(data):
    # 遍历 chosen 和 rejected
    for key in ["chosen", "rejected"]:
        for item in sample.get(key, []):
            if item.get("role") == "assistant":
                content = item.get("content", "").strip()
                if content == "":
                    print(json.dumps(sample, indent=2, ensure_ascii=False))
                    empty_cases.append((i, key))  # 记录第几个样本，在哪个key里

# 打印结果
if empty_cases:
    print("发现空的 assistant content：")
    for idx, key in empty_cases:
        print(f"样本 {idx} 的 {key} 里 assistant.content 为空")
else:
    print("没有发现空的 assistant content")
