# tasks/todo.md — Lab 22 Task Checklist

**Mục tiêu:** 100 pts core + 13 pts bonus = 113 pts  
**Deadline:** 23:59 ngày 2026-05-09  
**Cập nhật lần cuối:** 2026-05-08

---

## PHASE 0 — Setup (trước tất cả)

- [ ] **T0.1** Sao chép `.env.example` → `.env`
- [ ] **T0.2** Điền `OPENAI_API_KEY` vào `.env`
- [ ] **T0.3** Điền `HF_TOKEN` vào `.env` (cho Option B)
- [ ] **T0.4** Điền `WANDB_API_KEY` và `WANDB_PROJECT=lab22-dpo` vào `.env`
- [ ] **T0.5** Chạy `bash setup-laptop.sh`
- [ ] **T0.6** Chạy `make smoke` — phải exits 0
- [ ] **T0.7** Chụp `01-setup-gpu.png` → `submission/screenshots/`

**CHECKPOINT 0:** `make smoke` ✓

---

## PHASE 1 — Core Pipeline (100 pts)

### NB1 — SFT-mini (15 pts)
- [ ] **T1.1** Mở `notebooks/01_sft_mini.py` (hoặc `make sft`)
- [ ] **T1.2** Xác nhận config: T4, Qwen2.5-3B, SFT_SLICE=1000
- [ ] **T1.3** Chạy toàn bộ notebook — theo dõi loss
- [ ] **T1.4** Xác nhận `adapters/sft-mini/adapter_config.json` tồn tại (r:16, lora_alpha:32)
- [ ] **T1.5** Xác nhận `02-sft-loss.png` được lưu (loss giảm monotonic)
- [ ] **T1.6** Xác nhận sample generation print bằng tiếng Việt

**CHECKPOINT 1:** `ls adapters/sft-mini/adapter_config.json` ✓

### NB2 — Preference Data (10 pts)
- [ ] **T2.1** Chạy `notebooks/02_preference_data.py` (hoặc `make data`)
- [ ] **T2.2** Xác nhận 2000 pairs loaded
- [ ] **T2.3** Xác nhận 3 examples in đúng: token counts + `chosen ≠ rejected`
- [ ] **T2.4** Xác nhận `data/pref/train.parquet` có columns `prompt/chosen/rejected`
- [ ] **T2.5** Ghi chú fit%: ___% (phải ≥ 70%)

**CHECKPOINT 2:** `python -c "import pandas as pd; df=pd.read_parquet('data/pref/train.parquet'); print(len(df))"` → 2000 ✓

### [BONUS PREP] W&B Config (làm TRƯỚC NB3)
- [ ] **T5a.1** Mở `notebooks/03_dpo_train.py`
- [ ] **T5a.2** Đổi `report_to="none"` → `report_to="wandb"` trong DPOConfig (cell §2)

### NB3 — DPO Training (25 pts) ← QUAN TRỌNG NHẤT
- [ ] **T3.1** Chạy `notebooks/03_dpo_train.py` (hoặc `make dpo`) — ~30 phút
- [ ] **T3.2** Đọc kỹ failure-mode self-check output (cell §5a):
  - Kết quả: [ ] ✓ INTENDED / [ ] ⚠ LIKELIHOOD DISPLACEMENT / [ ] ✗ FAILURE
- [ ] **T3.3** Xác nhận plot có **2 curves riêng biệt** (chosen + rejected) KHÔNG CHỈ gap
- [ ] **T3.4** Xác nhận `03-dpo-reward-curves.png` được lưu
- [ ] **T3.5** Xác nhận `adapters/dpo/adapter_config.json` tồn tại
- [ ] **T3.6** Kiểm tra `dpo_metrics.json`: `end_reward_gap` > 0
- [ ] **T3.7** Ghi lại: chosen reward cuối = ___, rejected reward cuối = ___, gap = ___

**CHECKPOINT 3:** `python -c "import json; m=json.load(open('adapters/dpo/dpo_metrics.json')); print(m['end_reward_gap'])"` → số > 0 ✓

### NB4 — Qualitative Eval (10 pts)
- [ ] **T4.1** Xác nhận `OPENAI_API_KEY` trong `.env`
- [ ] **T4.2** Chạy `notebooks/04_compare_and_eval.py` (hoặc `make eval`)
- [ ] **T4.3** Xác nhận 8 responses từ SFT-only generated
- [ ] **T4.4** Xác nhận 8 responses từ SFT+DPO generated
- [ ] **T4.5** Xác nhận `04-side-by-side-table.png` có 8 rows + category column
- [ ] **T4.6** Xác nhận gpt-4o-mini judge chạy (không fallback manual)
- [ ] **T4.7** Ghi lại win/loss/tie: SFT wins ___, DPO wins ___, ties ___
- [ ] **T4.8** Chụp `05-judge-output.png` (≥ 3 judge verdicts, không có API key trong ảnh)

**CHECKPOINT 4:** `data/eval/judge_results.json` có 8 entries ✓

### NB5 — Merge + GGUF (10 pts)
- [ ] **T5.1** Chạy `notebooks/05_merge_deploy_gguf.py` (hoặc `make deploy`)
- [ ] **T5.2** Xác nhận `adapters/merged-fp16/` được tạo
- [ ] **T5.3** Xác nhận GGUF file tồn tại: `ls gguf/*.gguf`
- [ ] **T5.4** Kiểm tra size < 5 GB: `ls -lh gguf/*.gguf`  → ___ MB
- [ ] **T5.5** Xác nhận smoke prompt trả về ≥ 20 tokens tiếng Việt coherent
- [ ] **T5.6** Chụp `06-gguf-smoke.png` (phải thấy filename GGUF Q4_K_M trong load line)

**CHECKPOINT 5:** GGUF < 5 GB, smoke response coherent ✓

### NB6 — Benchmark (10 pts)
- [ ] **T6.1** Chạy `notebooks/06_benchmark.py` (hoặc `make bench`) — ~30 phút
- [ ] **T6.2** IFEval xong: SFT=___, DPO=___, Δ=___
- [ ] **T6.3** GSM8K xong: SFT=___, DPO=___, Δ=___
- [ ] **T6.4** MMLU xong: SFT=___, DPO=___, Δ=___
- [ ] **T6.5** AlpacaEval-lite xong (cần API key): DPO win-rate=___
- [ ] **T6.6** Xác nhận `data/eval/benchmark_results.json` có 4 benchmarks
- [ ] **T6.7** Xác nhận `07-benchmark-comparison.png` có deltas annotated

**CHECKPOINT 6:** `benchmark_results.json` có 4 keys, không có NaN ✓

---

## PHASE 2 — REFLECTION.md (22 pts)

- [ ] **T8.1** Mở `submission/REFLECTION.md`
- [ ] **T8.2** §1 Setup: điền bảng GPU/CUDA/model/dataset/cost
- [ ] **T8.3** §2 DPO results: điền bảng SFT-only vs SFT+DPO (từ `dpo_metrics.json`)
- [ ] **T8.4** §3 Reward curves: viết ≥ 100 từ — phân tích cả chosen VÀ rejected, reference deck §3.4
  - [ ] Mention "likelihood displacement" nếu applicable
  - [ ] Mô tả shape: flat đầu 100 steps rồi trend?
  - [ ] KL divergence cuối training
- [ ] **T8.5** §4 Qualitative: điền bảng 8 rows (từ `judge_results.json`)
  - [ ] Điền winner cho từng row (không để "SFT | DPO | tie" literal)
  - [ ] Điền win/loss/tie summary
- [ ] **T8.6** §5 β trade-off:
  - [ ] Nếu làm T7: điền bảng 3 rows số thật
  - [ ] Nếu không làm T7: viết hypothesis 3 câu
- [ ] **T8.7** §6 Personal reflection: viết ≥ 150 từ — 1 quyết định, 4 câu hỏi
  - [ ] Quyết định nào? (β, dataset, judge, T4 vs BigGPU...)
  - [ ] Alternative là gì?
  - [ ] Lý do chọn
  - [ ] Kết quả confirm hay surprise?
  - [ ] Thay đổi gì nếu làm lại?
- [ ] **T8.8** §7 Benchmark: viết ≥ 150 từ — reference deck §8.1 alignment tax
  - [ ] Benchmark nào tăng nhất?
  - [ ] GSM8K giảm không? → alignment tax?
  - [ ] MMLU flat hay thay đổi?
  - [ ] AlpacaEval-lite khớp với NB4 không?
- [ ] **T8.9** Đếm word count §3 (≥ 100): ___ từ ✓
- [ ] **T8.10** Đếm word count §6 (≥ 150): ___ từ ✓
- [ ] **T8.11** Đếm word count §7 (≥ 150): ___ từ ✓
- [ ] **T8.12** Bonus section: tick các checkbox đúng

**CHECKPOINT 7:** Không còn `<...>` placeholder trong REFLECTION.md ✓

---

## PHASE 3 — Bonus (nếu còn thời gian)

### β-sweep (+6 pts)
- [ ] **T7.1** Chạy `make beta-sweep` — ~90 phút
- [ ] **T7.2** Xác nhận 3 thư mục `adapters/dpo-b*/` đều có `dpo_metrics.json`
- [ ] **T7.3** Viết script plot β vs reward_gap (xem gợi ý trong plan.md T7)
- [ ] **T7.4** Lưu `bonus-beta-sweep.png` → `submission/screenshots/`
- [ ] **T7.5** Cập nhật REFLECTION §5 với số thật + ≥ 100 từ interpretation

### HF Hub push — Submission Option B (+5 pts)
- [ ] **T5b.1** `huggingface-cli login --token $HF_TOKEN`
- [ ] **T5b.2** `huggingface-cli upload 2A202600057-HoDacToan/lab22-dpo-vn ./adapters/dpo`
- [ ] **T5b.3** Tạo model card trên HF với base/dataset/hyperparameters/eval results
- [ ] **T5b.4** Cập nhật `README.md` GitHub để link đến HF model
- [ ] **T5b.5** Tick checkbox "Đã push lên HuggingFace Hub" trong REFLECTION Bonus

### W&B run link (+2 pts) — cần cấu hình T5a TRƯỚC NB3
- [ ] **T5a.3** Verify W&B run tồn tại tại `wandb.ai/<username>/lab22-dpo`
- [ ] **T5a.4** Set run to public
- [ ] **T5a.5** Copy link → paste vào REFLECTION Bonus section

---

## PHASE 4 — Verify + Submit

- [ ] **T9.1** Chạy `make verify` → phải exits 0
  - Nếu fail: đọc output, fix từng item, re-run
- [ ] **T9.2** Kiểm tra 7 screenshots tồn tại trong `submission/screenshots/`
- [ ] **T9.3** Đọc lại REFLECTION.md toàn bộ — không có placeholder
- [ ] **T9.4** `git add -A`
- [ ] **T9.5** `git commit -m "Lab 22 submission — Ho Dac Toan"`
- [ ] **T9.6** `git push -u origin main`
- [ ] **T9.7** Mở URL repo trong trình duyệt ẩn danh — xác nhận public
- [ ] **T9.8** Paste URL vào VinUni LMS Day-22 box trước 23:59 ngày 2026-05-09

---

## Tổng kết điểm dự kiến

| Hạng mục | Điểm | Trạng thái |
|---|---|---|
| NB1 (SFT-mini) | 15 | ⬜ |
| NB2 (Preference data) | 10 | ⬜ |
| NB3 (DPO training) | 25 | ⬜ |
| NB4 (Side-by-side eval) | 10 | ⬜ |
| NB5 (GGUF deploy) | 10 | ⬜ |
| NB6 (Benchmark) | 10 | ⬜ |
| REFLECTION.md | 22 | ⬜ |
| make verify pass | 3 | ⬜ |
| **Core subtotal** | **100** | |
| β-sweep bonus | +6 | ⬜ |
| HF Hub push bonus | +5 | ⬜ |
| W&B run link bonus | +2 | ⬜ |
| **Bonus subtotal** | **+13** | |
| **TỔNG** | **113** | |

> Cập nhật trạng thái: ⬜ = chưa làm · 🔄 = đang làm · ✅ = hoàn thành · ❌ = blocked
