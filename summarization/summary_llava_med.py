# %%
# Install first (if not already)
# pip install rouge-score


# %%
import torch
import pandas as pd
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
import time
from typing import List, Tuple, Optional
from rouge_score import rouge_scorer, scoring


# %%
# === 1) GPU Check ===
assert torch.cuda.is_available(), "Please enable a GPU runtime."
print("CUDA device:", torch.cuda.get_device_name(0))
torch.backends.cuda.matmul.allow_tf32 = True

# === 2) Model Choice (Quantized LLaVA-Med) ===
MODEL_ID = "chaoyinshe/llava-med-v1.5-mistral-7b-hf"

bnb_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# %%
# === 3) Load Model and Processor ===
print("Loading quantized LLaVA model...")
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_4bit,
    device_map="auto",
    low_cpu_mem_usage=True,
    attn_implementation=None,
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
print("Model loaded successfully.")

# === 4) Build Prompt Function (Text Only) ===
def build_text_prompt(processor, question: str):
    """Builds a text-only chat prompt compatible with LLaVA"""
    messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "apply_chat_template"):
        return processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        return f"[INST] {question} [/INST]"

# %%
# === 5) Inference Function (Text Only) ===
def generate_text_response(question, model, processor, max_new_tokens=256):
    # Prepare input prompt
    prompt = build_text_prompt(processor, question)
    inputs = processor(text=prompt, return_tensors="pt").to(model.device)

    # Generate output
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
        )

    # Decode output text
    decode_fn = getattr(processor, "decode", None) or processor.tokenizer.decode
    response = decode_fn(output_ids[0], skip_special_tokens=True)

    # Clean up (remove chat formatting if present)
    if "[/INST]" in response:
        response = response.split("[/INST]", 1)[-1].strip()

    return response.strip()

# %%
# === 6) Example Usage ===
if __name__ == "__main__":
    csv_path = "./data/LLaVA-Med/file3.csv"
    readable_output_file = "./data/LLaVA-Med/Results_LLaVA-Med_Summary_101125_1.txt" 
    results_readable = []

    # question = """You are a named entity recognition(NER) system. Extract and categorize the subject, treatment and effect from the given text. 
    # Generate the answer in this format. Format: "[Adverse event] associated with [Subject] a case of chronic renal failure in a patient [Treatment] AZ [Effect] acute hemorrhagic gastritis"
    # Text: " """

    question = """ You are an excellent clinical information extraction system. Summarise the adverse drug events and drugs causing that if mentioned in the provided text. An adverse drug event is any harmful or negative experience related to ongoing medication or treatment. Text: """

    no_of_entries_to_process = 50 ########

    df = pd.read_csv(csv_path)

    start = time.perf_counter()

    print(f"\nPrompt without Context: {question}")
    results_readable.append(f"\nPrompt without Context: {question}")
    aggregator = scoring.BootstrapAggregator()

    for index, row in df.iterrows():
        if index == no_of_entries_to_process:
            break

        quest_context = question + row['Preprocessed Posts']

        print(f"---------------------------------------")
        results_readable.append(f"---------------------------------------")
        
        print(f"\n🧠 Context: {row['Preprocessed Posts']}")
        results_readable.append(f"\n🧠 Context: {row['Preprocessed Posts']}")

        print(f"🪙 Gold Reference: {row['generated_summary']}")
        results_readable.append(f"🪙 Gold Reference: {row['generated_summary']}")

        answer = generate_text_response(quest_context, model, processor)
        print(f"💬 LLaVA Response: {answer}")
        results_readable.append(f"💬 LLaVA Response: {answer}")

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(row['generated_summary'], answer)

        # Add to bootstrap aggregator (aggregator expects dict of Scores)
        aggregator.add_scores(scores)
        
        for metric, score in scores.items():
            print(f"   {metric.upper():7s} | P={score.precision:.3f}  R={score.recall:.3f}  F1={score.fmeasure:.3f}")
            results_readable.append(f"   {metric.upper():7s} | P={score.precision:.3f}  R={score.recall:.3f}  F1={score.fmeasure:.3f}")

    print("-" * 60)
    results_readable.append("-" * 60)

    # Aggregate bootstrap results: print aggregated mid (mean) scores
    aggregated = aggregator.aggregate()
    print("\n🔹 Final Aggregate (bootstrap mid) ROUGE scores:")
    results_readable.append("\n🔹 Final Aggregate (bootstrap mid) ROUGE scores:")

    for metric, agg in aggregated.items():
        mid = agg.mid
        print(f"{metric.upper():7s} | P={mid.precision:.3f}  R={mid.recall:.3f}  F1={mid.fmeasure:.3f}")
        results_readable.append(f"{metric.upper():7s} | P={mid.precision:.3f}  R={mid.recall:.3f}  F1={mid.fmeasure:.3f}")

    final_avg_f1 = sum(aggregated[m].mid.fmeasure for m in aggregated) / len(aggregated)
    print("-" * 60)
    results_readable.append("-" * 60)
    print(f"⭐ Final Average F1 (ROUGE-1, ROUGE-2, ROUGE-L): {final_avg_f1:.3f}")
    results_readable.append(f"⭐ Final Average F1 (ROUGE-1, ROUGE-2, ROUGE-L): {final_avg_f1:.3f}")

    end = time.perf_counter()

    print("-" * 60)
    results_readable.append("-" * 60)
    print(f"Execution time: {end - start:.6f} seconds")
    results_readable.append(f"Execution time: {end - start:.6f} seconds")

    # Write per-image results to CSV
    with open(readable_output_file, "w", encoding="utf-8") as f:
      f.write("\n".join(results_readable))

    print(f"\n--- Readable results written to: {readable_output_file} ---")


