# tasks/plan.md — Lab 22 Implementation Plan

**Sinh viên:** Ho Dac Toan (2A202600057)  
**Ngày:** 2026-05-08  
**Tier:** T4 | Judge: gpt-4o-mini | Bonus: β-sweep (+6) + HF Hub (+5) + W&B (+2)  
**Mục tiêu tổng:** 100 pts core + 13 pts bonus = 113 pts

---

## Dependency Graph

```
[T0] Environment Setup
        │
        ▼
[T1] NB1 — SFT-mini (15 pts)
        │
   ┌────┴────┐
   ▼         ▼
[T2] NB2   [T5a] Configure W&B (+2 bonus)
   │              (trước khi NB3 chạy)
   └────┬────┘
        ▼
[T3] NB3 — DPO Training (25 pts)  ←── critical path
        │
   ┌────┼────────┬─────────────┐
   ▼    ▼        ▼             ▼
[T4]  [T5]    [T6]          [T7]
NB4   NB5     NB6           β-sweep
eval  GGUF    bench         (+6 bonus)
10pts 10pts   10pts
   │    │
   │    └──► [T5b] HF Hub push (+5 bonus)
   │
   └────┬────────┘
        ▼
[T8] REFLECTION.md (22 pts)
        │
        ▼
[T9] Verify + Submit
```

**Critical path:** T0 → T1 → T2 → T3 → T4/T5/T6 → T8 → T9  
**Bonus path:** T3 → T7 (β-sweep); T5 → T5b (HF push); T5a trước T3 (W&B)

---

## Ước tính thời gian tổng

| Phase | Tasks | Thời gian |
|---|---|---|
| Setup + NB1 + NB2 | T0 + T1 + T2 | ~17 phút |
| NB3 DPO training | T3 | ~30 phút |
| NB4 + NB5 (song song logic) | T4 + T5 | ~15 phút |
| NB6 Benchmark | T6 | ~30 phút |
| REFLECTION + Verify + Submit | T8 + T9 | ~30 phút |
| **Core subtotal** | | **~122 phút (~2 giờ)** |
| β-sweep (bonus) | T7 | ~90 phút |
| HF Hub push + W&B | T5a + T5b | ~10 phút |
| **Tổng kể cả bonus** | | **~222 phút (~3.7 giờ)** |

> **Khuyến nghị:** hoàn thành core pipeline (T0→T9) trước, sau đó mới làm bonus.

---

## T0 — Environment Setup

**Mục tiêu:** GPU hoạt động, dependencies cài xong, `.env` có đủ keys.

**Các bước:**
1. Clone repo (nếu chưa có)
2. `cp .env.example .env` — điền các keys thật:
   - `COMPUTE_TIER=T4`
   - `OPENAI_API_KEY=sk-...`
   - `HF_TOKEN=hf_...` (cho Submission Option B)
   - `WANDB_API_KEY=...` (cho bonus W&B)
   - `WANDB_PROJECT=lab22-dpo`
3. `bash setup-laptop.sh` — cài venv + deps + cuda probe (~5 phút)
4. `make smoke` — 2-step training verify

**Acceptance criteria:**
- [ ] `make smoke` exits 0
- [ ] `nvidia-smi` hoặc `torch.cuda.get_device_name()` hiện đúng GPU + VRAM
- [ ] Screenshot `01-setup-gpu.png` được lưu vào `submission/screenshots/`

**Verification:** `python -c "import torch; print(torch.cuda.get_device_name(0))"` → tên GPU

---

## T1 — NB1: SFT-mini Checkpoint

**Mục tiêu:** Build Qwen2.5-3B SFT checkpoint trên 1k VN Alpaca, 1 epoch. Tạo base cho DPO.

**Input:** GPU ready, `.env` có `COMPUTE_TIER=T4`  
**Output:** `adapters/sft-mini/adapter_config.json`, `submission/screenshots/02-sft-loss.png`

**Các bước:**
1. Mở [notebooks/01_sft_mini.py](../notebooks/01_sft_mini.py) trong Jupyter (hoặc `make sft`)
2. Kiểm tra cell §0 config: `COMPUTE_TIER=T4`, `BASE_MODEL=unsloth/Qwen2.5-3B-bnb-4bit`, `SFT_SLICE=1000`
3. Chạy toàn bộ notebook — theo dõi loss tại `logging_steps=10`
4. Xác nhận cell §3a tạo loss curve file `submission/screenshots/02-sft-loss.png`
5. Xác nhận cell §4 in ra sample generation (sanity check)

**Acceptance criteria:**
- [ ] `adapters/sft-mini/adapter_config.json` có `lora_alpha: 32, r: 16`
- [ ] Loss curve giảm monotonic (không có spike lớn sau bước 200+)
- [ ] Final train loss ≤ 1.5 (gợi ý; không bắt buộc nhưng nên đạt)
- [ ] Ít nhất 1 sample generation được in, trả lời bằng tiếng Việt

**Traps phổ biến:**
- `tokenizer.pad_token` chưa set → cell §1 đã handle (`eos_token` fallback)
- OOM → `COMPUTE_TIER` sai; hoặc restart Jupyter kernel trước khi chạy

**CHECKPOINT T1:** `ls adapters/sft-mini/` phải có `adapter_config.json` và `adapter_model.safetensors`

---

## T2 — NB2: Preference Data Prep

**Mục tiêu:** Load 2k UltraFeedback pairs, format `prompt/chosen/rejected`, save Parquet.

**Input:** `adapters/sft-mini/` tồn tại (tokenizer từ NB1)  
**Output:** `data/pref/train.parquet` (2000 rows), `data/pref/eval.parquet` (50 rows)

**Các bước:**
1. Chạy [notebooks/02_preference_data.py](../notebooks/02_preference_data.py) (hoặc `make data`)
2. Cell §2 load `argilla/ultrafeedback-binarized-preferences-cleaned` slice 2000
3. Cell §3 format với chat template Qwen2.5 → `prompt/chosen/rejected`
4. Cell §3a: kiểm tra 3 examples — xem token counts, `chosen ≠ rejected` cho mỗi row
5. Cell §3b: xem length distribution — mục tiêu ≥ 80% pairs fit trong `MAX_LEN=512`
6. Cell §4: save Parquet

**Acceptance criteria:**
- [ ] `data/pref/train.parquet` có ≥ 1900 rows (tối thiểu; 2000 lý tưởng)
- [ ] Columns: `prompt`, `chosen`, `rejected`
- [ ] `chosen ≠ rejected` cho tất cả 3 examples được in
- [ ] Fit % được in trong cell §3b (phải ≥ 70%, cảnh báo nếu < 80%)

**CHECKPOINT T2:** `python -c "import pandas as pd; df=pd.read_parquet('data/pref/train.parquet'); print(len(df), df.columns.tolist())"`

---

## T5a — Configure W&B (làm TRƯỚC T3, bonus +2)

**Mục tiêu:** Kích hoạt Weights & Biases logging cho NB3 trước khi train.

**Điều kiện:** `WANDB_API_KEY` đã có trong `.env`, `WANDB_PROJECT=lab22-dpo`

**Thay đổi trong [notebooks/03_dpo_train.py](../notebooks/03_dpo_train.py):**

Tìm dòng này trong cell §2 (DPOConfig):
```python
report_to="none",
```
Đổi thành:
```python
report_to="wandb",
```

**Verify:** sau khi NB3 chạy, `wandb.ai` sẽ có run mới trong project `lab22-dpo`.  
Copy public link → paste vào REFLECTION.md section Bonus.

**Acceptance criteria:**
- [ ] `wandb` run xuất hiện tại `wandb.ai/<username>/lab22-dpo`
- [ ] Run có training curves visible (reward_gap, chosen_rewards, rejected_rewards, loss)
- [ ] Link được paste vào REFLECTION.md Bonus section

---

## T3 — NB3: DPO Training ← Critical Path (25 pts)

**Mục tiêu:** Train DPO adapter trên SFT-mini + 2k UltraFeedback. Plot dual reward curves.

**Input:** `adapters/sft-mini/`, `data/pref/train.parquet`  
**Output:** `adapters/dpo/`, `submission/screenshots/03-dpo-reward-curves.png`, `adapters/dpo/dpo_metrics.json`

**Các bước:**
1. (Nếu làm W&B bonus) Đã đổi `report_to="none"` → `"wandb"` ở T5a
2. Chạy [notebooks/03_dpo_train.py](../notebooks/03_dpo_train.py) (hoặc `make dpo`) — ~30 phút
3. Cell §4: theo dõi DPO loss (sẽ giảm từ ~0.6 xuống ~0.3-0.4 ở 1 epoch)
4. Cell §5: **QUAN TRỌNG** — đọc kỹ output của failure-mode self-check:
   - `✓ INTENDED` → ghi vào REFLECTION §3 là "classic DPO success"
   - `⚠ LIKELIHOOD DISPLACEMENT` → ghi vào REFLECTION §3 là hiện tượng bình thường, reference deck §3.4
   - `✗ FAILURE` → debug trước khi tiếp tục (xem risk register)
5. Cell §5 phải tạo plot có **2 curves riêng biệt** (chosen + rejected) CỘNG gap plot
6. Cell §6: save adapter và `dpo_metrics.json`

**Acceptance criteria:**
- [ ] `adapters/dpo/adapter_config.json` tồn tại, khác với `sft-mini/adapter_config.json`
- [ ] `adapters/dpo/dpo_metrics.json` có `end_reward_gap > 0`
- [ ] `03-dpo-reward-curves.png` có **CẢ HAI** chosen_rewards VÀ rejected_rewards được plot riêng biệt + gap
- [ ] Plot có legend, xlabel ("Training step"), ylabel, title

**⚠ ĐIỂM MẤT CAO NHẤT:** Nếu plot chỉ có gap mà không có 2 curves riêng → mất 10 pts!

**Traps phổ biến:**
- Colab T4 OOM tại step 1: `gradient_accumulation_steps` 8→16, `max_length` 512→384
- `chosen_rewards` flat sau 500 steps: `beta` 0.1→0.05 hoặc `lr` 5e-7→1e-6
- TRL version mismatch: pin `trl>=0.12,<0.20`

**CHECKPOINT T3:** 
```bash
python -c "import json; m=json.load(open('adapters/dpo/dpo_metrics.json')); print(f'gap={m[\"end_reward_gap\"]:.3f}')"
```
Phải in số > 0.

---

## T4 — NB4: Qualitative Comparison (10 pts)

**Mục tiêu:** 8 prompts × {SFT, SFT+DPO}, judge bằng gpt-4o-mini, win/loss/tie summary.

**Input:** `adapters/sft-mini/`, `adapters/dpo/`, `OPENAI_API_KEY` trong `.env`  
**Output:** `submission/screenshots/04-side-by-side-table.png`, `05-judge-output.png`, `data/eval/judge_results.json`

**Các bước:**
1. Xác nhận `.env` có `OPENAI_API_KEY` và `JUDGE_MODEL=gpt-4o-mini`
2. Chạy [notebooks/04_compare_and_eval.py](../notebooks/04_compare_and_eval.py) (hoặc `make eval`)
3. Cell §2: generate 8 responses từ SFT-only (~5 phút)
4. Cell §3: generate 8 responses từ SFT+DPO (~5 phút)
5. Cell §4: render bảng markdown + save `04-side-by-side-table.png`
6. Cell §5: chạy gpt-4o-mini judge (tự động vì có OPENAI_API_KEY)
7. Cell §6: in win/loss/tie summary
8. **Chụp thêm `05-judge-output.png`:** screenshot cell output của cell §5 cho ≥ 3 prompts

**Acceptance criteria:**
- [ ] `04-side-by-side-table.png` có đủ 8 rows với cột category (helpfulness/safety)
- [ ] `data/eval/judge_results.json` có 8 entries với `winner` field (A/B/tie)
- [ ] Win/loss/tie summary được in rõ ràng (Overall + Helpfulness + Safety breakdown)
- [ ] `05-judge-output.png` chụp ≥ 3 judge verdicts (không bao gồm API key trong ảnh)

**Chú ý bảo mật:** crop ảnh để không thấy `sk-...` trong output cell.

---

## T5 — NB5: Merge + GGUF + Deploy (10 pts)

**Mục tiêu:** Merge adapter vào base, export GGUF Q4_K_M, smoke test llama.cpp.

**Input:** `adapters/sft-mini/`, `adapters/dpo/`  
**Output:** `gguf/lab22-dpo-Q4_K_M.gguf`, `submission/screenshots/06-gguf-smoke.png`, `data/eval/deploy_meta.json`

**Các bước:**
1. Chạy [notebooks/05_merge_deploy_gguf.py](../notebooks/05_merge_deploy_gguf.py) (hoặc `make deploy`)
2. Cell §1: load base + SFT adapter + DPO adapter
3. Cell §2: `save_pretrained_merged("merged_16bit")` → `adapters/merged-fp16/` (~5 phút)
4. Cell §3: `save_pretrained_gguf(quantization="q4_k_m")` → `gguf/` (~2 phút)
5. Cell §4: `Llama(model_path=..., n_gpu_layers=-1)` + smoke prompt tiếng Việt
6. **Chụp `06-gguf-smoke.png`:** toàn bộ cell §4a output: filename GGUF trong load line + response

**Acceptance criteria:**
- [ ] `gguf/*.gguf` tồn tại với size < 5 GB
- [ ] Smoke response ≥ 20 tokens coherent tiếng Việt (không phải gibberish)
- [ ] `06-gguf-smoke.png` hiện filename GGUF (`Q4_K_M`) trong load line
- [ ] `data/eval/deploy_meta.json` có `gguf_size_mb`, `smoke_response`

**Trap:** `merge_and_unload()` fail với "tied weights" → `del model.config.tie_word_embeddings` trước khi merge

**CHECKPOINT T5:** `ls -lh gguf/*.gguf` — phải thấy file < 5 GB

---

## T5b — HuggingFace Hub Push (bonus +5, Submission Option B)

**Mục tiêu:** Push DPO adapter lên HF Hub với model card hoàn chỉnh.

**Điều kiện:** T5 đã hoàn thành (GGUF tồn tại), `HF_TOKEN` trong `.env`

**Các bước:**
1. Login HF: `huggingface-cli login --token $HF_TOKEN`
2. Push adapter: `huggingface-cli upload 2A202600057-HoDacToan/lab22-dpo-vn ./adapters/dpo`
3. Tạo model card tại HF Hub với nội dung:
   ```markdown
   ---
   base_model: unsloth/Qwen2.5-3B-bnb-4bit
   datasets:
     - 5CD-AI/Vietnamese-alpaca-cleaned
     - argilla/ultrafeedback-binarized-preferences-cleaned
   language: vi
   tags: [dpo, alignment, vietnamese, lora]
   ---
   # lab22-dpo-vn
   DPO-aligned Qwen2.5-3B adapter — Lab 22 AICB-P2T3 VinUniversity
   Base: unsloth/Qwen2.5-3B-bnb-4bit
   SFT: 1k Vietnamese Alpaca, 1 epoch, LoRA r=16 α=32
   DPO: 2k UltraFeedback, beta=0.1, lr=5e-7, 1 epoch
   ```
4. Update `README.md` của GitHub repo để link đến HF model

**Acceptance criteria:**
- [ ] HF Hub có repo `2A202600057-HoDacToan/lab22-dpo-vn` với `adapter_config.json`
- [ ] Model card có base model, dataset, hyperparameters, evaluation results từ NB6
- [ ] GitHub README link đến HF model

---

## T6 — NB6: Quantitative Benchmark (10 pts)

**Mục tiêu:** IFEval + GSM8K + MMLU(500) + AlpacaEval-lite trên SFT-only vs SFT+DPO. 4-bar chart.

**Input:** `adapters/sft-mini/`, `adapters/dpo/`, `OPENAI_API_KEY` (cho AlpacaEval-lite)  
**Output:** `data/eval/benchmark_results.json`, `submission/screenshots/07-benchmark-comparison.png`

**Các bước:**
1. Chạy [notebooks/06_benchmark.py](../notebooks/06_benchmark.py) (hoặc `make bench`) — ~30 phút
2. Cell §2: IFEval trên cả 2 adapters (~8 phút)
3. Cell §3: GSM8K trên cả 2 adapters (~8 phút)
4. Cell §4: MMLU sampled 500 trên cả 2 adapters (~8 phút)
5. Cell §5: AlpacaEval-lite 100 prompts với gpt-4o-mini judge (~5 phút, cần API key)
6. Cell §6: 4-bar comparison plot với deltas annotated
7. Cell §7: save `benchmark_results.json`

**Acceptance criteria:**
- [ ] `benchmark_results.json` có `metrics` với 4 keys: IFEval, GSM8K, MMLU, AlpacaEval-lite
- [ ] Mỗi benchmark có `sft` và `dpo` scores (không phải NaN)
- [ ] `07-benchmark-comparison.png` có 4 cặp bars với delta annotated (Δ=+x.xxx hoặc Δ=-x.xxx)
- [ ] Plot có legend, title, y-axis label

**Nếu NB6 timeout:** giảm `LIMIT_MMLU=200` và `LIMIT_GSM8K=200` trong cell §0.

**Nếu AlpacaEval-lite skip** (không có API key): acceptaable — IFEval/GSM8K/MMLU vẫn đủ điểm.

---

## T7 — β-sweep Mini Experiment (bonus +6)

**Mục tiêu:** So sánh DPO với β ∈ {0.05, 0.1, 0.5}. Plot reward gap + win-rate vs β.

**Điều kiện:** T3 đã xong (NB3 chạy được). **Làm sau khi core pipeline (T0→T9) hoàn thành.**

**Các bước:**
1. `make beta-sweep` — tự động chạy NB3 3 lần với các giá trị β khác nhau (~90 phút)
2. Mỗi run lưu vào `adapters/dpo-b{0.05,0.1,0.5}/dpo_metrics.json`
3. Tự viết script plot β vs reward_gap (gợi ý từ NB3 vibe-coding callout):
   ```python
   import json, matplotlib.pyplot as plt
   from pathlib import Path
   results = []
   for d in sorted(Path("adapters").glob("dpo-b*")):
       m = json.loads((d / "dpo_metrics.json").read_text())
       results.append((m["beta"], m["end_reward_gap"]))
   betas, gaps = zip(*results)
   plt.plot(betas, gaps, marker="o")
   plt.xlabel("β"); plt.ylabel("Reward gap (end)")
   plt.title("β vs Reward Gap")
   plt.savefig("submission/screenshots/bonus-beta-sweep.png")
   ```
4. Điền REFLECTION §5 với bảng 3 rows + interpretation ≥ 100 từ

**Hypothesis trước khi chạy (think-hard zone):**
- β=0.05 (aggressive): DPO tự do hơn → gap lớn hơn nhưng có thể length hacking
- β=0.1 (default): balanced
- β=0.5 (conservative): KL penalty mạnh → gap nhỏ hơn → outputs ít thay đổi so với SFT

**Acceptance criteria:**
- [ ] `adapters/dpo-b0.05/`, `adapters/dpo-b0.1/`, `adapters/dpo-b0.5/` đều có `dpo_metrics.json`
- [ ] `bonus-beta-sweep.png` tồn tại với β vs reward_gap plot
- [ ] REFLECTION §5 bảng 3 rows được điền số thật
- [ ] REFLECTION §5 có ≥ 100 từ interpretation, reference deck §3.3

---

## T8 — REFLECTION.md (22 pts)

**Mục tiêu:** Điền đầy đủ 7 sections với số liệu thật từ các notebook đã chạy.

**Input:** Tất cả screenshot, `dpo_metrics.json`, `benchmark_results.json`, `judge_results.json`

**Checklist từng section:**

| Section | Điền gì | Nguồn số liệu | Min words |
|---|---|---|---|
| §1 Setup | GPU model, CUDA, base model, datasets, cost | Cell §0 output của mỗi NB | — |
| §2 DPO results | Bảng SFT-only vs SFT+DPO metrics | `dpo_metrics.json`, cell §4/§5 NB3 | — |
| §3 Reward curves | Phân tích chosen + rejected trajectories | `03-dpo-reward-curves.png`, NB3 cell §5a output | ≥ 100 từ |
| §4 Qualitative | Bảng 8 examples + winner, win/loss/tie summary | `judge_results.json`, NB4 output | — |
| §5 β trade-off | Bảng 3 rows (nếu làm sweep) hoặc hypothesis | `dpo_metrics.json` các runs | — |
| §6 Personal reflection | 1 quyết định + 4 câu hỏi: alternative, lý do, kết quả, thay đổi gì | Tự viết | ≥ 150 từ |
| §7 Benchmark interpretation | Bảng 4 benchmarks + delta, alignment tax analysis | `benchmark_results.json`, `07-benchmark-comparison.png` | ≥ 150 từ |

**Hướng dẫn §3** (reward curves analysis — 5 pts):
- Mô tả shape của chosen_rewards: tăng, giảm, hay flat?
- Mô tả shape của rejected_rewards: giảm, tăng, hay flat?
- Kết luận: "classic DPO success" hay "likelihood displacement" (deck §3.4)?
- KL divergence cuối training là bao nhiêu? (từ log NB3)
- Tại sao reward gap quan trọng hơn là chỉ nhìn chosen reward?

**Hướng dẫn §7** (benchmark — 2 pts):
- Benchmark nào tăng nhiều nhất? Tại sao? (IFEval tăng = chat-tuning works)
- GSM8K có giảm không? (Alignment tax — deck §8.1)
- MMLU thay đổi bao nhiêu? (Kiến thức nền có bị ảnh hưởng không?)
- AlpacaEval-lite win-rate so với NB4 judge results — khớp hay khác? Tại sao?

**Acceptance criteria:**
- [ ] Tất cả 7 sections có nội dung (không còn placeholder text)
- [ ] §3 ≥ 100 từ, mention "likelihood displacement" hoặc "deck §3.4"
- [ ] §6 ≥ 150 từ, answer rõ ràng 4 câu hỏi trong template
- [ ] §7 ≥ 150 từ, mention "alignment tax" hoặc "deck §8.1"
- [ ] Không có số `<...>` placeholder
- [ ] §4 bảng 8 rows có `winner` điền thật (không phải "SFT | DPO | tie" literal)

---

## T9 — Verify + Submit

**Mục tiêu:** Gatekeeper pass, push GitHub, submit LMS.

**Các bước:**

1. **Run gatekeeper:**
   ```bash
   make verify
   ```
   Nếu fail, đọc output và fix từng item trước khi tiếp tục.

2. **Kiểm tra screenshots checklist:**
   ```
   submission/screenshots/01-setup-gpu.png    ✓
   submission/screenshots/02-sft-loss.png     ✓
   submission/screenshots/03-dpo-reward-curves.png  ✓ (dual curve!)
   submission/screenshots/04-side-by-side-table.png ✓
   submission/screenshots/05-judge-output.png       ✓
   submission/screenshots/06-gguf-smoke.png         ✓
   submission/screenshots/07-benchmark-comparison.png ✓
   submission/screenshots/bonus-beta-sweep.png      ✓ (nếu làm T7)
   ```

3. **Push GitHub:**
   ```bash
   git add -A
   git commit -m "Lab 22 submission — Ho Dac Toan"
   git push -u origin main
   ```

4. **Verify public:** truy cập repo URL trong browser ẩn danh — phải xem được file

5. **Submit LMS:** paste public GitHub URL vào VinUni LMS Day-22 box

**Acceptance criteria:**
- [ ] `make verify` exits 0
- [ ] Repo public tại `github.com/<username>/Day22-Track3-DPO-Alignment-Lab`
- [ ] URL đã được paste vào LMS trước deadline 23:59 ngày 2026-05-09

---

## Risk Register

| Rủi ro | Xác suất | Mitigation | Deadline impact |
|---|---|---|---|
| Colab T4 OOM tại NB3 | Cao | `grad_accum` 8→16; `max_length` 512→384 | +15 phút |
| `chosen_rewards` flat sau 500 steps | Trung bình | `beta` 0.1→0.05 hoặc `lr` 5e-7→1e-6; re-run NB3 | +30 phút |
| NB6 timeout > 90 phút | Trung bình | `LIMIT_MMLU=200`, `LIMIT_GSM8K=200` | giảm benchmark quality |
| OpenAI rate limit NB4 | Thấp | Manual rubric mode (no points lost) | 0 |
| HF Hub CLI auth fail | Thấp | Re-login; kiểm tra `HF_TOKEN` expiry | +10 phút |
| β-sweep overruns deadline | Trung bình | Chỉ làm sau core pipeline done; skip nếu < 2 giờ còn lại | mất +6 pts bonus |

---

## Thứ tự ưu tiên khi thiếu thời gian

1. T0 + T1 + T2 + T3 → **50 pts** (không thể thiếu)
2. T8 (REFLECTION §3 + §6 + §7) → **22 pts** (nhiều điểm nhất per hour)
3. T4 (NB4 table) → **10 pts** (chỉ 15 phút)
4. T9 (verify + submit) → **3 pts** (5 phút, không nên skip)
5. T5 (NB5 GGUF) → **10 pts** (15 phút)
6. T6 (NB6 benchmark) → **10 pts** (30 phút — skip nếu hết thời gian)
7. T7 β-sweep + T5a W&B + T5b HF push → **+13 pts bonus** (làm cuối cùng)
