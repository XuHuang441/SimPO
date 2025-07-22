# import sys
# print("PYTHONPATH includes:\n", sys.path)
import logging
import sys
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import torch
import transformers
from accelerate import Accelerator
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, HfArgumentParser, AutoTokenizer, PreTrainedTokenizerBase

# 假设这些是你项目中的辅助工具和配置
from alignment import DataArguments, H4ArgumentParser, ModelArguments
from alignment.data import is_openai_format
from inpo_scripts.run_inpo import get_batch_logps  # 从我们之前的脚本导入logps计算函数

from trl.trainer.utils import DPODataCollatorWithPadding
import pprint

logger = logging.getLogger(__name__)


# =====================================================================================
# NEW: 专为预计算设计的参数类
# =====================================================================================
@dataclass
class PrecomputationArgs:
    input_dataset_name: str = field(metadata={"help": "Path or name of the input dataset."})
    output_dataset_path: str = field(metadata={"help": "Path to save the final augmented dataset."})
    reference_model_path: str = field(metadata={"help": "Path to the initial SFT/reference model."})
    history_model_paths: Optional[List[str]] = field(
        default_factory=list,
        metadata={"help": "A list of paths to historical models for logps computation.", "nargs": "*"},
    )
    per_device_batch_size: int = field(default=4, metadata={"help": "Batch size per device for precomputation."})
    torch_dtype: str = field(default="bfloat16",
                             metadata={"help": "Torch dtype for loading models (e.g., bfloat16, float16)."})


class DPODataCollator:
    """
    用于处理 prompt + chosen / rejected 数据结构，提取 assistant 回复并拼接。
    输入数据样例：
        {
            "prompt": "question...",
            "chosen": [{"role": "user", ...}, {"role": "assistant", "content": "..."}],
            "rejected": [{"role": "user", ...}, {"role": "assistant", "content": "..."}]
        }
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int = 2048, label_pad_token_id: int = -100):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts = [f["prompt"] for f in features]
        chosen_responses = [f["chosen"][-1]["content"] for f in features]
        rejected_responses = [f["rejected"][-1]["content"] for f in features]

        chosen_texts = [p + c for p, c in zip(prompts, chosen_responses)]
        rejected_texts = [p + r for p, r in zip(prompts, rejected_responses)]

        # === 1. Tokenize separately WITHOUT padding ===
        chosen = self.tokenizer(chosen_texts, padding=False, truncation=True, max_length=self.max_length)
        rejected = self.tokenizer(rejected_texts, padding=False, truncation=True, max_length=self.max_length)

        # === 2. 统一长度 padding 到两者最大值 ===
        max_len = max(
            max(len(x) for x in chosen["input_ids"]),
            max(len(x) for x in rejected["input_ids"])
        )
        chosen = self.tokenizer.pad(chosen, padding="max_length", max_length=max_len, return_tensors="pt")
        rejected = self.tokenizer.pad(rejected, padding="max_length", max_length=max_len, return_tensors="pt")

        prompt_encodings = self.tokenizer(prompts, padding=False, truncation=True, max_length=self.max_length)
        prompt_lens = [len(x) for x in prompt_encodings["input_ids"]]

        chosen_labels = chosen["input_ids"].clone()
        rejected_labels = rejected["input_ids"].clone()
        for i, l in enumerate(prompt_lens):
            chosen_labels[i, :l] = self.label_pad_token_id
            rejected_labels[i, :l] = self.label_pad_token_id

        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
            "rejected_labels": rejected_labels,
        }


def compute_and_add_logps(
        dataset: DatasetDict,
        model_path: str,
        tokenizer: transformers.PreTrainedTokenizerBase,
        args: PrecomputationArgs,
        accelerator: Accelerator,
        column_prefix: str,
        max_length: int = 2048,
) -> DatasetDict:
    """
    使用指定模型计算 chosen 和 rejected 的 logps，并添加到数据集。
    使用自定义 DataCollator 自动处理拼接与 masking。
    """
    logger.info(f"--- Processing model: {model_path} for columns with prefix: '{column_prefix}' ---")

    # 加载模型
    model_dtype = getattr(torch, args.torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=model_dtype,
        use_cache=False,
    ).eval()
    model = accelerator.prepare_model(model)

    # 初始化 Collator
    data_collator = DPODataCollator(tokenizer, max_length=max_length)

    for split in dataset.keys():
        split_dataset = dataset[split]
        logger.info(f"Computing logps for '{split}' split...")

        dataloader = DataLoader(split_dataset, batch_size=args.per_device_batch_size,
                                shuffle=False, collate_fn=data_collator)
        dataloader = accelerator.prepare(dataloader)

        all_chosen_logps, all_rejected_logps = [], []

        for batch in tqdm(dataloader, desc=f"Computing '{column_prefix}' logps for {split}"):
            with torch.no_grad():
                # 将输入送入模型
                input_ids = torch.cat([batch["chosen_input_ids"], batch["rejected_input_ids"]], dim=0)
                attention_mask = torch.cat([batch["chosen_attention_mask"], batch["rejected_attention_mask"]], dim=0)
                labels = torch.cat([batch["chosen_labels"], batch["rejected_labels"]], dim=0)

                input_ids = input_ids.to(accelerator.device)
                attention_mask = attention_mask.to(accelerator.device)
                labels = labels.to(accelerator.device)

                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

                all_logps = get_batch_logps(logits, labels, label_pad_token_id=-100)
                bsz = batch["chosen_input_ids"].size(0)
                chosen_logps, rejected_logps = all_logps[:bsz], all_logps[bsz:]

            chosen_logps, rejected_logps = accelerator.gather_for_metrics((chosen_logps, rejected_logps))
            all_chosen_logps.append(chosen_logps.cpu())
            all_rejected_logps.append(rejected_logps.cpu())

        dataset[split] = split_dataset.add_column(f"{column_prefix}_chosen_logps", torch.cat(all_chosen_logps).numpy())
        dataset[split] = split_dataset.add_column(f"{column_prefix}_rejected_logps",
                                                  torch.cat(all_rejected_logps).numpy())
        logger.info(f"Added '{column_prefix}_*' logps to '{split}' split.")

    del model
    accelerator.free_memory()
    torch.cuda.empty_cache()
    logger.info(f"--- Finished processing model: {model_path}. Memory cleaned. ---")
    return dataset


def main():
    parser = HfArgumentParser(PrecomputationArgs)
    args = parser.parse_args_into_dataclasses()[0]

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(args.reference_model_path)  # 用参考模型路径初始化tokenizer即可

    # 1. 加载原始数据集
    logger.info(f"Loading initial dataset from: {args.input_dataset_name}")
    if args.input_dataset_name.endswith((".json", ".jsonl")):
        current_dataset = DatasetDict.from_json({"train": args.input_dataset_name})
    else:
        current_dataset = load_dataset(args.input_dataset_name)

    # 2. 计算参考模型的 logps
    current_dataset = compute_and_add_logps(
        dataset=current_dataset,
        model_path=args.reference_model_path,
        tokenizer=tokenizer,
        args=args,
        accelerator=accelerator,
        column_prefix="reference",
    )

    # 3. 遍历历史模型列表，依次计算 logps
    if args.history_model_paths:
        for i, model_path in enumerate(args.history_model_paths):
            current_dataset = compute_and_add_logps(
                dataset=current_dataset,
                model_path=model_path,
                tokenizer=tokenizer,
                args=args,
                accelerator=accelerator,
                column_prefix=f"history{i}",
            )

    # 4. 保存最终的、包含所有 logps 的数据集
    if accelerator.is_main_process:
        logger.info(f"All computations finished. Final columns: {current_dataset['train'].column_names}")
        logger.info(f"Saving final augmented dataset to: {args.output_dataset_path}")
        current_dataset.save_to_disk(args.output_dataset_path)
        logger.info("Script finished successfully.")


if __name__ == "__main__":
    main()