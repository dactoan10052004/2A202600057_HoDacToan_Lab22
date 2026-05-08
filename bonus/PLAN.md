# Implementation Plan — Tư Vấn Tuyển Sinh Đại Học

> Estimate: 6–8 giờ total. Có thể thực hiện 1 người.

---

## Phase 0 — Bootstrap (30 phút)

**Goal:** Setup folder, verify pipeline tái sử dụng được từ NB3.

Tasks:
- [ ] Copy `notebooks/03_dpo_train.py` → `bonus/train.py`, swap dataset path
- [ ] Verify `unsloth/Qwen2.5-3B-bnb-4bit` còn cached trên Colab Drive
- [ ] Tạo `bonus/data/` directory

---

## Phase 1 — Data Construction (3–4 giờ)

**Goal:** 200 preference pairs chất lượng, từ domain judgment thật — không dùng UltraFeedback.

### 1a. Thu thập prompts (1.5 giờ)

Nguồn câu hỏi thực tế (không cần crawl — ghi tay từ memory + observation):

| Nguồn | Loại câu hỏi | Target |
|---|---|---|
| Facebook group "Tư vấn tuyển sinh" | Điểm chuẩn, nguyện vọng | 80 prompts |
| Diễn đàn hocmai.vn / hoc247.net | Tổ hợp môn, ngành nghề | 60 prompts |
| Câu hỏi phổ biến tự construct | Edge cases, boundary prompts | 60 prompts |

**Categories:**
- Điểm chuẩn / xét tuyển cơ chế (50 prompts)
- Chọn ngành / trường (40 prompts)
- Tổ hợp môn (30 prompts)
- Phương thức xét tuyển (30 prompts)
- Boundary / out-of-scope (50 prompts — quan trọng cho alignment)

Output: `bonus/data/prompts.jsonl`
```json
{"id": 1, "prompt": "...", "category": "diem_chuan", "is_boundary": false}
```

### 1b. Construct preference pairs (2 giờ)

Với mỗi prompt, viết tay:
- `chosen`: thông tin nền + nguồn + tone ấm + KHÔNG dự đoán số liệu
- `rejected`: một trong 3 failure mode:
  - Hallucinate điểm chuẩn cụ thể
  - Từ chối lạnh không có ích
  - Trả lời generic English-heavy

**Ví dụ pair:**
```json
{
  "prompt": "Em 24.5 điểm thi THPT, có đăng ký CNTT Bách Khoa HN được không?",
  "chosen": "Điểm 24.5 nằm trong vùng cạnh tranh của CNTT BKHN trong những năm gần đây. Tuy nhiên điểm chuẩn thay đổi mỗi năm tùy số lượng thí sinh và chỉ tiêu — mình không thể dự đoán chính xác. Bạn có thể tra điểm chuẩn các năm trước tại tuyensinh.hust.edu.vn để tự so sánh. Đừng quên đăng ký nhiều nguyện vọng khác nhau làm lưới an toàn nhé!",
  "rejected": "Với 24.5 điểm tổ hợp A00, bạn có thể đỗ CNTT Bách Khoa vì điểm chuẩn năm nay dự kiến khoảng 24-25 điểm."
}
```

Output: `bonus/data/pairs.parquet` (200 rows, columns: prompt, chosen, rejected, category)

---

## Phase 2 — DPO Training (1–1.5 giờ trên Colab T4)

**Goal:** Fine-tune adapter trên 200 preference pairs.

### Hyperparameters (reuse từ NB3, giảm scale)
```python
COMPUTE_TIER = "T4"
MODEL_NAME   = "unsloth/Qwen2.5-3B-bnb-4bit"
BETA         = 0.1
LR           = 5e-7
EPOCHS       = 1
MAX_SEQ_LEN  = 1024   # prompt tuyển sinh ngắn hơn alpaca
```

### train.py structure
```
1. Load model (reuse unsloth FastLanguageModel pattern từ NB3)
2. Load bonus/data/pairs.parquet
3. Format thành DPO Dataset (prompt / chosen / rejected)
4. DPOTrainer fit
5. Save adapter → bonus/adapters/dpo-bonus/
6. Save metrics → bonus/adapters/dpo-bonus/dpo_metrics.json
```

**Success criteria:** `end_reward_gap > 0` (chosen reward > rejected reward)

---

## Phase 3 — Evaluation (45 phút)

**Goal:** Verify model không hallucinate số liệu, boundary cases được handle đúng.

### Test set: 20 prompts
- 10 benign-but-specific: câu hỏi thông thường, model NÊN trả lời với resource
- 10 boundary-crossing: dự đoán cụ thể, tư vấn cá nhân, model phải soft-refuse + handoff

Metric: precision/recall trên 2 tập — tính thủ công (20 prompts, dễ label).

Output: `bonus/demo/5-samples.md` — 5 prompt đại diện + output SFT-only vs SFT+DPO

---

## Phase 4 — Demo + Model Card (1 giờ)

### demo/serve.py (~60 dòng)
```python
import gradio as gr
from llama_cpp import Llama  # hoặc transformers nếu không có GGUF

llm = Llama(model_path="bonus/gguf/tuvan-tuyen-sinh-Q4_K_M.gguf", n_ctx=1024)

SYSTEM_PROMPT = """Bạn là trợ lý tư vấn tuyển sinh đại học Việt Nam.
Nguyên tắc: cung cấp thông tin nền + nguồn chính thức.
KHÔNG dự đoán điểm chuẩn cụ thể. KHÔNG đưa ra quyết định thay học sinh."""

def chat(message, history):
    # format prompt, call llm, return response
    ...

gr.ChatInterface(chat, title="Tư Vấn Tuyển Sinh 🎓").launch()
```

### MODEL-CARD.md
```
- Model name: tuvan-tuyen-sinh-v0
- Base: unsloth/Qwen2.5-3B-bnb-4bit
- DPO data: 200 pairs (self-constructed, VN university admissions domain)
- What it does: answers VN university admissions questions
- What it does NOT do: predict scores, replace official counselors
- Known limitations: data staleness, North Vietnam bias
- Vibe coding log: [ghi prompt AI hiệu quả nhất / fail nhất]
```

---

## Timeline

| Phase | Thời gian | Deliverable |
|---|---|---|
| Phase 0 | 30 phút | train.py skeleton ready |
| Phase 1a | 1.5 giờ | 200 prompts in prompts.jsonl |
| Phase 1b | 2 giờ | 200 pairs in pairs.parquet |
| Phase 2 | 1–1.5 giờ | dpo-bonus adapter + metrics |
| Phase 3 | 45 phút | 5-samples.md + test set results |
| Phase 4 | 1 giờ | serve.py + MODEL-CARD.md |
| **Total** | **~7 giờ** | **Full bonus submission** |

---

## Risk & Mitigation

| Risk | Mitigation |
|---|---|
| 200 pairs quá ít để thấy reward gap | Nếu gap ≤ 0: giải thích trong MODEL-CARD (scale limitation), giá trị vẫn là data construction + deploy |
| Colab session reset giữa Phase 2 | Drive checkpoint sau khi save adapter (reuse pattern từ NB3) |
| GGUF export fail (transformers 5.5 compat) | Đã có fix pattern từ NB5 — reuse cell 98-105 |
| Data bias (North VN only) | Document rõ trong Honest Limitations, không overclaim |
