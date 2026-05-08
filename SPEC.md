# SPEC.md — Lab 22 DPO/ORPO Alignment

**Sinh viên:** Ho Dac Toan (2A202600057)  
**Ngày tạo:** 2026-05-08  
**Tier:** T4 (Free Colab / laptop GPU ≥ 12 GB)  
**Mục tiêu điểm:** 100 pts core + 13 pts bonus (β-sweep +6, HF Hub +5, W&B +2) = **113 pts**  
**Submission:** Option B — Professional (push adapters lên HuggingFace Hub)  
**Deadline:** 23:59 ngày 2026-05-09

---

## 1. Objective

### Mục tiêu chính

Build một DPO-aligned Vietnamese language model end-to-end trên Free Colab T4 / laptop GPU:

1. **SFT-mini checkpoint** — Tái tạo Lab 21 inline: Qwen2.5-3B-bnb-4bit + LoRA r=16 + 1k VN Alpaca, 1 epoch
2. **Preference data** — Load UltraFeedback (2k pairs T4), format `prompt/chosen/rejected`, save Parquet
3. **DPO training** — TRL `DPOTrainer(beta=0.1, lr=5e-7)`, plot reward curves (chosen + rejected riêng biệt)
4. **Qualitative eval** — 8 prompts × {SFT, SFT+DPO}, judge bằng `gpt-4o-mini` (OpenAI API)
5. **GGUF export** — Merge adapter → GGUF Q4_K_M < 5 GB → llama.cpp smoke test
6. **Quantitative benchmark** — IFEval / GSM8K / MMLU(500) / AlpacaEval-lite + 4-bar chart

### Target users

Sinh viên AICB-P2T3 VinUniversity A20 cohort 2026 — bài nộp cá nhân.

### Acceptance criteria (definition of done)

- [ ] `make verify` exit 0
- [ ] 6 screenshots bắt buộc trong `submission/screenshots/`
- [ ] `submission/REFLECTION.md` 7 sections đầy đủ, §3 ≥ 100 từ, §6 ≥ 150 từ, §7 ≥ 150 từ
- [ ] `adapters/dpo/` được push lên HuggingFace Hub với model card
- [ ] Repo public trên GitHub, URL submit vào VinUni LMS

---

## 2. Configuration

### Biến môi trường (.env)

```env
# Tier
COMPUTE_TIER=T4

# DPO hyperparameters (deck §5.2)
DPO_BETA=0.1
DPO_LR=5e-7
DPO_EPOCHS=1

# Judge — OpenAI (NB4)
OPENAI_API_KEY=sk-...          # điền key thật
JUDGE_MODEL=gpt-4o-mini

# HuggingFace (Submission Option B)
HF_TOKEN=hf_...                # điền token thật
HF_REPO=2A202600057-HoDacToan/lab22-dpo-vn

# Weights & Biases (bonus +2)
WANDB_API_KEY=...               # điền key thật
WANDB_PROJECT=lab22-dpo
```

### Commands theo thứ tự thực hiện

```bash
# 1. Setup (một lần)
bash setup-laptop.sh
cp .env.example .env            # rồi điền các key ở trên

# 2. Smoke test — xác nhận GPU + imports OK
make smoke

# 3. Full pipeline (NB1 → NB6)
make pipeline

# 4. Bonus β-sweep (sau khi pipeline xong)
make beta-sweep

# 5. Push lên HuggingFace Hub (Option B)
huggingface-cli upload 2A202600057-HoDacToan/lab22-dpo-vn ./adapters/dpo

# 6. Pre-submission gatekeeper
make verify

# 7. Commit + push lên GitHub
git add -A
git commit -m "Lab 22 submission — Ho Dac Toan"
git push -u origin main
```

### Make targets theo từng notebook

| Command | Notebook | Thời gian (T4) |
|---|---|---|
| `make sft` | NB1 — SFT-mini | ~10 phút |
| `make data` | NB2 — Preference data | ~2 phút |
| `make dpo` | NB3 — DPO training | ~30 phút |
| `make eval` | NB4 — Compare + judge | ~5 phút |
| `make deploy` | NB5 — Merge + GGUF | ~5 phút |
| `make bench` | NB6 — Benchmark | ~30 phút |
| `make beta-sweep` | NB3 × 3 (β sweep) | ~90 phút |

---

## 3. Project Structure

```
2A202600057_HoDacToan_Lab22/
├── SPEC.md                        # ← file này
├── README.md
├── VIBE-CODING.md
├── rubric.md
├── Makefile
├── .env                           # KHÔNG commit — chứa API keys
├── .env.example
├── requirements.txt
├── requirements-biggpu.txt
├── pyproject.toml
│
├── notebooks/                     # Jupytext .py — source of truth
│   ├── 01_sft_mini.py             # NB1: SFT checkpoint
│   ├── 02_preference_data.py      # NB2: UltraFeedback prep
│   ├── 03_dpo_train.py            # NB3: DPO training + reward curves
│   ├── 04_compare_and_eval.py     # NB4: side-by-side + GPT-4o-mini judge
│   ├── 05_merge_deploy_gguf.py    # NB5: merge + GGUF Q4_K_M
│   └── 06_benchmark.py            # NB6: IFEval/GSM8K/MMLU/AlpacaEval-lite
│
├── colab/                         # Colab-launchable mirrors
│   ├── Lab22_DPO_T4.ipynb
│   └── Lab22_DPO_BigGPU.ipynb
│
├── scripts/
│   ├── prepare_preference_data.py
│   ├── train_dpo.py
│   ├── eval_judge.py              # GPT-4o-mini / Claude judge
│   ├── merge_and_gguf.py
│   └── verify.py                  # pre-submission gatekeeper
│
├── adapters/                      # gitignored — populated by training
│   ├── sft-mini/                  # NB1 output: LoRA r=16, α=32
│   ├── dpo/                       # NB3 output: DPO LoRA + dpo_metrics.json
│   ├── dpo-b0.05/                 # β-sweep output (bonus)
│   ├── dpo-b0.1/                  # β-sweep output (bonus)
│   └── dpo-b0.5/                  # β-sweep output (bonus)
│
├── data/                          # gitignored — populated by NB2
│   ├── pref/train.parquet         # UltraFeedback formatted
│   └── eval/
│       └── benchmark_results.json # NB6 output
│
├── gguf/                          # NB5 output
│   └── lab22-dpo-Q4_K_M.gguf     # < 5 GB
│
└── submission/
    ├── REFLECTION.md              # 7 sections — bắt buộc điền
    └── screenshots/
        ├── 01-setup-gpu.png       # nvidia-smi + VRAM
        ├── 02-sft-loss.png        # NB1 loss curve (monotonic decrease)
        ├── 03-dpo-reward-curves.png  # NB3 dual-curve: chosen + rejected + gap
        ├── 04-side-by-side-table.png # NB4 ≥ 8 prompts × 2 models
        ├── 05-judge-output.png    # GPT-4o-mini verdict ≥ 3 prompts
        ├── 06-gguf-smoke.png      # NB5 llama.cpp coherent VN output
        ├── 07-benchmark-comparison.png # NB6 4-bar chart with deltas
        └── bonus-beta-sweep.png   # β-sweep chart (bonus +6)
```

### Artifacts quan trọng theo notebook

| Notebook | Input cần có | Output tạo ra |
|---|---|---|
| NB1 | GPU + `COMPUTE_TIER` | `adapters/sft-mini/`, `02-sft-loss.png` |
| NB2 | NB1 done | `data/pref/train.parquet` |
| NB3 | NB1 + NB2 done | `adapters/dpo/`, `03-dpo-reward-curves.png`, `dpo_metrics.json` |
| NB4 | NB3 done + `OPENAI_API_KEY` | `04-side-by-side-table.png`, `05-judge-output.png` |
| NB5 | NB3 done | `gguf/lab22-dpo-Q4_K_M.gguf`, `06-gguf-smoke.png` |
| NB6 | NB1 + NB3 done | `benchmark_results.json`, `07-benchmark-comparison.png` |

---

## 4. Code Style

### Nguyên tắc chung

- **Giữ nguyên `# %%` cell markers** trong `.py` Jupytext files — Jupytext dùng để sync với `.ipynb`
- **Không sửa hyperparameters** khỏi giá trị deck §5.2 nếu không ghi rõ lý do (beta=0.1, lr=5e-7)
- **Đặt tên screenshot đúng convention** trong `submission/screenshots/README.md`
- **Không commit `.env`** — chỉ commit `.env.example`

### Conventions cho Jupytext notebooks

```python
# Cell marker — bắt buộc giữ nguyên format
# %%

# Markdown cell
# %% [markdown]
# ## Tiêu đề

# Config từ env — pattern chuẩn
import os
COMPUTE_TIER = os.environ.get("COMPUTE_TIER", "T4").upper()
```

### Reward curve plotting — yêu cầu bắt buộc

Phải plot **cả 2 curves riêng biệt** (NB3, chiếm 10 pts):

```python
# ĐỦ TIÊU CHUẨN (10 pts)
axes[0].plot(steps, chosen_rewards, label="chosen reward")  # curve 1
axes[0].plot(steps, rejected_rewards, label="rejected reward")  # curve 2
axes[1].plot(steps, gap, label="reward gap")  # gap = chosen - rejected

# KHÔNG ĐỦ (mất 10 pts)
plt.plot(steps, gap)  # chỉ plot gap
```

### REFLECTION.md — word count requirements

| Section | Yêu cầu tối thiểu | Điểm |
|---|---|---|
| §3 Reward curves | ≥ 100 từ + reference deck §3.4 | 5 pts |
| §6 Personal reflection | ≥ 150 từ + 4 câu hỏi bắt buộc | từ 15 pts |
| §7 Benchmark interpretation | ≥ 150 từ + reference deck §8.1 | 2 pts |

---

## 5. Testing Strategy

### Pre-flight check (trước khi chạy full pipeline)

```bash
make smoke      # 2-step training verify — phải xanh trước khi commit time
```

**Smoke test kiểm tra:**
- GPU available (`torch.cuda.is_available()`)
- Unsloth + TRL imports OK
- Model load không OOM
- 2-step training không crash

### Checkpoints trong pipeline

| Sau bước | Kiểm tra | Fail = làm gì |
|---|---|---|
| NB1 | `adapters/sft-mini/adapter_config.json` tồn tại; loss ≤ 1.5 ở cuối | Re-run; xem bảng Common Gotchas |
| NB2 | `data/pref/train.parquet` có 2000 rows; cột `chosen ≠ rejected` | Kiểm tra UltraFeedback filter logic |
| NB3 | `reward_gap > 0` ở cuối; `adapters/dpo/dpo_metrics.json` tồn tại | Nếu gap âm: đổi beta 0.1→0.05; nếu flat: tăng lr |
| NB4 | ≥ 8 rows trong table; judge response không phải `null` | Kiểm tra `OPENAI_API_KEY` trong `.env` |
| NB5 | GGUF < 5 GB; smoke prompt returns ≥ 20 tokens tiếng Việt | Nếu merge fail: xoá `tie_word_embeddings` trước merge |
| NB6 | `benchmark_results.json` có 4 keys; không có NaN values | Giảm `LIMIT_MMLU` nếu timeout |

### Pre-submission gatekeeper

```bash
make verify     # Phải exit 0 trước khi push
```

`verify.py` kiểm tra toàn bộ artifacts required. Output sẽ liệt kê mọi item thiếu.

### Likelihood displacement self-check (NB3)

NB3 có cell §5a tự động phân loại:
- ✓ `INTENDED` — chosen UP, gap positive → classic DPO success
- ⚠ `LIKELIHOOD DISPLACEMENT` — chosen DOWN, rejected faster → document in §3
- ✗ `FAILURE` — gap NEGATIVE → fix data/config, re-run

---

## 6. Boundaries

### Luôn làm (Always)

- **Giữ output cells trong `.ipynb`** khi commit — grader cần thấy kết quả chạy
- **Chụp màn hình đủ 6 ảnh** trước khi push — verify.py sẽ fail nếu thiếu
- **Điền REFLECTION.md bằng số liệu thật** của mình, không copy từ README
- **Chạy `make verify`** trước mỗi lần push
- **Giữ repo public** cho đến khi điểm được công bố

### Hỏi trước (Ask first)

- Thay đổi hyperparameters khỏi deck §5.2 defaults (beta, lr, epochs) — phải ghi lý do rõ trong REFLECTION §6
- Dùng dataset khác ngoài UltraFeedback / VN Alpaca
- Chạy trên BigGPU tier nếu đã bắt đầu với T4 (artifacts sẽ không tương thích)
- Giảm `PREF_SLICE` xuống dưới 1000 (có thể ảnh hưởng reward gap)

### Không làm (Never)

- **Không commit `.env`** — chứa API keys, token; chỉ commit `.env.example`
- **Không push API keys lên GitHub** — crop screenshots nếu key xuất hiện
- **Không dùng `--no-verify` khi commit** — bypass hook có thể hide broken state
- **Không submit khi repo private** — grader không xem được = 0 điểm
- **Không plot chỉ reward gap mà bỏ qua chosen/rejected riêng biệt** (mất 10 pts NB3)
- **Không suppress OOM error bằng cách giảm batch mà không ghi rõ** — ghi thay đổi config vào REFLECTION §1

---

## 7. Bonus Scope

### β-sweep (+6 pts)

**Mục tiêu:** So sánh DPO behavior với β ∈ {0.05, 0.1, 0.5}

```bash
make beta-sweep
# → chạy NB3 3 lần, lưu vào adapters/dpo-b{0.05,0.1,0.5}/
# → mỗi run lưu dpo_metrics.json riêng
```

**Deliverable:**
- `bonus-beta-sweep.png` — chart reward gap + win-rate vs β
- REFLECTION §5 — điền bảng 3 rows + interpretation ≥ 100 từ

**Think-hard zone:** Predict trước khi nhìn kết quả — β cao → conservative (KL penalty mạnh) → gap nhỏ hơn; β thấp → aggressive → gap lớn hơn nhưng có thể hallucinate. Hypothesis này có khớp với deck §3.3 không?

### HuggingFace Hub push (+5 pts)

```bash
huggingface-cli login            # dùng HF_TOKEN từ .env
huggingface-cli upload 2A202600057-HoDacToan/lab22-dpo-vn ./adapters/dpo
```

**Model card phải có:**
- Base model: `unsloth/Qwen2.5-3B-bnb-4bit`
- Dataset: `argilla/ultrafeedback-binarized-preferences-cleaned` (2k pairs)
- Training: DPO, beta=0.1, lr=5e-7, 1 epoch
- Evaluation results (từ NB6 benchmark_results.json)

**Sau khi push:** update `README.md` của repo GitHub để link đến HF model.

### W&B run link (+2 pts)

Thêm vào `.env`:
```env
WANDB_API_KEY=...
WANDB_PROJECT=lab22-dpo
```

Trong NB3, đổi `report_to="none"` → `report_to="wandb"` trong `DPOConfig`.  
**Deliverable:** Public W&B link visible trong REFLECTION Bonus section.

---

## 8. Risk Register

| Rủi ro | Xác suất | Tác động | Mitigation |
|---|---|---|---|
| OOM tại DPO step 1 (Colab T4) | Cao | Block NB3 | Tăng `grad_accum` 8→16; giảm `max_length` 512→384 |
| `chosen_rewards` không tăng sau 500 steps | Trung bình | Mất 10 pts NB3 | Giảm beta 0.1→0.05 hoặc tăng lr 5e-7→1e-6 |
| NB6 timeout > 90 phút trên T4 | Trung bình | Không có benchmark | Giảm `LIMIT_MMLU` 500→200, `LIMIT_GSM8K` 500→200 |
| OpenAI API rate limit (NB4) | Thấp | Judge fallback | Dùng manual rubric mode nếu bị rate limit |
| GGUF merge fail "tied weights" | Thấp | Block NB5 | `del model.config.tie_word_embeddings` trước `merge_and_unload()` |
| Deadline miss do β-sweep quá dài | Trung bình | Mất +6 bonus | Bắt đầu beta-sweep sau khi core pipeline xong và còn > 90 phút |

---

*SPEC này được tạo ngày 2026-05-08 dựa trên rubric.md, README.md, và lựa chọn cấu hình của sinh viên.*
