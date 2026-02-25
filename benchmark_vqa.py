# benchmark_vqa.py — Evaluación baseline LLaVA vs Fine-tuned LLaVA (QLoRA)
"""
Compara accuracy VQA-v2 entre:
  - Baseline: LLaVA zero-shot (greedy)
  - Fine-tuned: LLaVA + LoRA adapter (greedy, misma instancia)

Usa un solo modelo en memoria y aplica/desactiva el adapter.

Uso:
  python benchmark_vqa.py --n_samples 20
  python benchmark_vqa.py --n_samples 20 --adapter outputs/lora_adapter
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from transformers import BitsAndBytesConfig, LlavaNextForConditionalGeneration, LlavaNextProcessor
from peft import PeftModel

from data.vqa_loader import VQADataset
from src.config import MODEL_ID, QUANTIZATION, TORCH_DTYPE, MAX_MEMORY
from src.symbol_detector import SymbolDetector

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def load_model(adapter_path=None):
    """Carga UN solo modelo LLaVA. Si adapter_path, aplica LoRA adapter."""
    print("🔧 Cargando modelo base...")
    bnb_config = BitsAndBytesConfig(**QUANTIZATION)
    processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        torch_dtype=TORCH_DTYPE,
        device_map="auto",
        max_memory=MAX_MEMORY,
        low_cpu_mem_usage=True,
    )

    has_adapter = False
    if adapter_path and Path(adapter_path).exists():
        print(f"🔌 Cargando LoRA adapter desde {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        has_adapter = True
        print("✅ Adapter cargado (enable/disable para comparar)")

    return model, processor, has_adapter


def inference(model, processor, image, question, max_tokens=30):
    """Inferencia estándar (greedy)."""
    prompt = f"USER: <image>\n{question}\nAnswer with a single word or short phrase. ASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Benchmark VQA-v2")
    parser.add_argument("--n_samples", type=int, default=20, help="Nº muestras")
    parser.add_argument("--adapter", type=str, default=None, help="Path al LoRA adapter")
    parser.add_argument("--output", type=str, default="outputs/benchmark_results.csv")
    args = parser.parse_args()

    # ── Cargar modelo (1 sola instancia) ───────────────────────────────
    model, processor, has_adapter = load_model(adapter_path=args.adapter)
    detector = SymbolDetector(model)

    # ── Dataset (eval split) ───────────────────────────────────────────
    dataset = VQADataset(
        split="val",
        max_samples=args.n_samples,
        train_ratio=0.8,
        is_train_split=False,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # ── Benchmark ──────────────────────────────────────────────────────
    results = []
    accs_base = []
    accs_ft = []
    scs_scores = []

    print(f"\n{'='*80}")
    print(f"  BENCHMARK VQA-v2: {len(dataset)} muestras (eval split)")
    print(f"  Modelo: {MODEL_ID}")
    if has_adapter:
        print(f"  Adapter: {args.adapter}")
    print(f"{'='*80}\n")

    for i in range(len(dataset)):
        sample = dataset[i]
        image = sample["image"]
        question = sample["question"]
        gt = sample["answer"]
        all_answers = sample["all_answers"]

        print(f"\n── {i+1}/{len(dataset)} ──────────────────────────")
        print(f"   Q: {question}")
        print(f"   GT: {gt}")

        # ── Baseline (sin adapter) ────────────────────────────────
        if has_adapter:
            model.disable_adapter_layers()

        t0 = time.time()
        ans_base = inference(model, processor, image, question)
        t_base = (time.time() - t0) * 1000

        acc_base = VQADataset.vqa_accuracy(ans_base, all_answers)
        accs_base.append(acc_base)
        print(f"   Baseline:   {ans_base} (acc={acc_base:.2f}, {t_base:.0f}ms)")

        # ── Fine-tuned (con adapter) ──────────────────────────────
        acc_ft_val = 0.0
        ans_ft = ""
        t_ft = 0.0
        if has_adapter:
            model.enable_adapter_layers()
            t0 = time.time()
            ans_ft = inference(model, processor, image, question)
            t_ft = (time.time() - t0) * 1000

            acc_ft_val = VQADataset.vqa_accuracy(ans_ft, all_answers)
            accs_ft.append(acc_ft_val)
            print(f"   Fine-tuned: {ans_ft} (acc={acc_ft_val:.2f}, {t_ft:.0f}ms)")

        # ── SCS (M2) ─────────────────────────────────────────────
        inputs = processor(
            text=f"USER: <image>\n{question}\nAnswer briefly. ASSISTANT:",
            images=image, return_tensors="pt"
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        symbols, _, variance = detector.extract_symbols(
            inputs["input_ids"],
            pixel_values=inputs.get("pixel_values"),
            image_sizes=inputs.get("image_sizes"),
        )
        n_clusters = len(set(symbols[symbols >= 0]))
        scs_scores.append(n_clusters)

        torch.cuda.empty_cache()

        results.append({
            "idx": i,
            "question": question,
            "gt_answer": gt,
            "baseline_answer": ans_base,
            "baseline_acc": acc_base,
            "finetuned_answer": ans_ft if has_adapter else "",
            "finetuned_acc": acc_ft_val,
            "n_clusters": n_clusters,
            "variance_pca": variance,
        })

    # ── Resumen ────────────────────────────────────────────────────────
    n = len(results)
    avg_base = sum(accs_base) / max(n, 1)
    avg_clusters = sum(scs_scores) / max(n, 1)

    print(f"\n{'='*80}")
    print(f"  RESULTADOS ({n} muestras)")
    print(f"{'='*80}")
    print(f"  {'Métrica':<25} {'Baseline':>10}", end="")

    if has_adapter:
        avg_ft = sum(accs_ft) / max(n, 1)
        print(f" {'Fine-tuned':>12}")
        print(f"  {'─'*25} {'─'*10} {'─'*12}")
        print(f"  {'Accuracy media':<25} {avg_base:>10.3f} {avg_ft:>12.3f}")
        delta = avg_ft - avg_base
        icon = '↑' if delta > 0 else '↓' if delta < 0 else '='
        print(f"\n  Δ Accuracy: {delta:+.3f} ({icon})")
    else:
        print()
        print(f"  {'─'*25} {'─'*10}")
        print(f"  {'Accuracy media':<25} {avg_base:>10.3f}")

    print(f"  {'Clusters medio':<25} {avg_clusters:>10.1f}")
    print(f"{'='*80}")

    # ── CSV + JSON ─────────────────────────────────────────────────────
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    summary = {
        "n_samples": n,
        "model": MODEL_ID,
        "adapter": args.adapter or "none",
        "baseline_accuracy": round(avg_base, 4),
        "finetuned_accuracy": round(avg_ft, 4) if has_adapter else None,
        "avg_clusters": round(avg_clusters, 1),
    }
    json_path = Path(args.output).with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n📊 CSV: {args.output}")
    print(f"📄 JSON: {json_path}")

    detector.remove_hooks()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
