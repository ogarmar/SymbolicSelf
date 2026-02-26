# test/test_symbolic_vqa.py — Test end-to-end: VQA con SymbolicSelf
"""
Ejecuta el pipeline SymbolicSelf completo sobre ejemplos VQA-v2 reales
usando el dataset de HuggingFace lmms-lab/vqav2.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from datasets import load_dataset

from src.model_loader import load_model_sync
from src.symbol_detector import SymbolDetector
from src.m1_self_polish import SelfPolishCore


def main():
    print("🔧 Cargando modelo para test VQA end-to-end...")

    model, processor = load_model_sync()

    detector = SymbolDetector(model)

    # Cargar 5 ejemplos de VQA-v2
    ds = load_dataset("lmms-lab/vqav2", split="validation[:5]")
    templates = ["Be precise.", "Clarify details.", "Complete answer."]

    for i, example in enumerate(ds):
        question = example["question"]
        image = example["image"]
        ground_truth = example["answer"]

        print(f"\n{'='*50}")
        print(f"🖼️ {i+1}: {question[:60]}")

        # ── Baseline (sin refinamiento) ────────────────────────────────
        prompt = f"USER: <image>\n{question} ASSISTANT:"
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            baseline_ids = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )

        new_tokens = baseline_ids[0][inputs["input_ids"].shape[-1]:]
        baseline = processor.decode(new_tokens, skip_special_tokens=True).strip()
        print(f"   Baseline: {baseline}")

        # ── Symbolic Polish (3 variantes refinadas) ────────────────────
        best_scs = 0.0
        best_response = baseline

        baseline_ids_tok = processor.tokenizer(baseline, return_tensors="pt").input_ids.to(model.device)
        baseline_symbols, _, _ = detector.extract_symbols(baseline_ids_tok)

        for template in templates:
            refine_prompt = f"USER: <image>\n{question} Refine: {template} {baseline} ASSISTANT:"
            refine_inputs = processor(text=refine_prompt, images=image, return_tensors="pt")
            refine_inputs = {k: v.to(model.device) for k, v in refine_inputs.items()}

            with torch.no_grad():
                refined_ids = model.generate(
                    **refine_inputs,
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )

            new_ids = refined_ids[0][refine_inputs["input_ids"].shape[-1]:]
            refined = processor.decode(new_ids, skip_special_tokens=True).strip()

            # Calcular SCS real
            var_ids = processor.tokenizer(refined, return_tensors="pt").input_ids.to(model.device)
            var_symbols, _, _ = detector.extract_symbols(var_ids)

            if len(var_symbols) > 0 and len(baseline_symbols) > 0:
                scs, metrics = detector.compute_scs(var_symbols, baseline_symbols)
            else:
                scs = 0.0

            if scs > best_scs:
                best_scs = scs
                best_response = refined

        print(f"   🏆 Symbolic SCS={best_scs:.3f}: {best_response}")
        print(f"   GT: {ground_truth}")

        torch.cuda.empty_cache()

    detector.remove_hooks()
    print(f"\n{'='*50}")
    print("🎉 Test VQA end-to-end completado.")


if __name__ == "__main__":
    main()
