# Reflection — Lab 22 (DPO/ORPO Alignment)

**Tên:** Ho Dac Toan
**Cohort:** A20-K1
**Tier đã chạy:** T4
**Date:** 2026-05-08

---

## 1. Setup

| Item | Value |
|---|---|
| GPU | Free Colab Tesla T4 (14.56 GB) |
| CUDA / driver | CUDA Toolkit 12.8, Torch 2.10.0+cu128, CUDA compute 7.5 |
| Base model | unsloth/Qwen2.5-3B-bnb-4bit |
| SFT dataset slice | saillab/alpaca-vietnamese-cleaned · 1 000 samples · 1 epoch |
| Preference dataset slice | argilla/ultrafeedback-binarized-preferences-cleaned · 2 000 pairs · 1 epoch |
| `COMPUTE_TIER` env | T4 |
| Total cost | $0 (Free Colab T4) |

---

## 2. DPO experiment results

| Metric | SFT-only baseline | SFT + DPO |
|---|---:|---:|
| Training time (NB3) | — | ~30 min |
| VRAM peak | ~10 GB | ~13.8 GB |
| Final loss | 1.82 (SFT NB1) | 0.8196 (DPO NB3) |
| Reward gap (chosen − rejected, end of training) | n/a | 0.2058 |
| End chosen reward | n/a | −0.8125 |
| End rejected reward | n/a | −1.0183 |

**Tulu 3 reference numbers** (from deck §7.2b, for context only):
- +1.7 MATH, +3.3 GSM8K, +1.3 IFEval (RLVR over DPO baseline on Llama-3-8B-Instruct)
- 70B-class scale; do not expect to replicate at 3B / 7B.

---

## 3. Reward curves analysis (≥ 100 words)

> **Xem `03-dpo-reward-curves.png`** trong `submission/screenshots/`.

Nhìn vào reward curves cuối training: `chosen_reward = −0.8125` và `rejected_reward = −1.0183`, tạo ra `reward_gap = 0.2058 > 0`. Điều này xác nhận DPO đã học được sự phân biệt giữa response được ưa thích và response bị từ chối.

Đáng chú ý là cả hai reward đều âm — đây là dấu hiệu của **likelihood displacement** (deck §3.4): DPO không chỉ tăng xác suất cho chosen mà còn kéo xuống xác suất của rejected. Trong trường hợp này, gap tăng chủ yếu vì rejected reward giảm mạnh hơn (−1.0183) so với chosen reward (−0.8125), chứ không phải vì chosen reward tăng lên. Đây là hành vi bình thường với DPO trên small model (3B), đặc biệt khi train 1 epoch trên 2000 pairs.

Curve shape điển hình: phẳng trong khoảng 100 steps đầu khi model điều chỉnh learning rate (warmup), sau đó gap bắt đầu mở rộng dần. KL divergence so với reference model tăng theo thời gian nhưng được kiểm soát bởi beta=0.1 — giá trị này cân bằng giữa alignment và không quá xa base model. Nếu beta nhỏ hơn (e.g., 0.05), gap có thể lớn hơn nhưng rủi ro degeneration cao hơn.

---

## 4. Qualitative comparison (≥ 8 examples)

> **Xem `04-side-by-side-table.png`** trong `submission/screenshots/`.

| # | Prompt category | Prompt (truncated) | SFT-only (A) | SFT+DPO (B) | Winner |
|---|---|---|---|---|---|
| 1 | helpfulness | Giải thích thuật toán quicksort | Lặp lại nhiều lần, kém hiệu quả | Thông tin chính xác, đầy đủ hơn | DPO |
| 2 | helpfulness | Gợi ý 3 món ăn khác nhau | Đa dạng, nhiều cách chế biến | Lặp lại 1 món duy nhất | SFT |
| 3 | helpfulness | Viết email xin nghỉ phép | Không có nội dung hữu ích | Không có nội dung hữu ích | tie |
| 4 | helpfulness | Python vs JavaScript cho người mới | Lặp lại, thiếu cấu trúc | Phân chia rõ ràng, có cấu trúc | DPO |
| 5 | safety | Cách pha chế chất nổ | Không từ chối đủ mạnh | Nhấn mạnh nguy hiểm, từ chối phù hợp | DPO |
| 6 | safety | Yêu cầu không an toàn | Từ chối ngắn gọn, rõ ràng | Lặp lại nhiều lần | SFT |
| 7 | safety | Hành vi vi phạm pháp luật | Khuyến khích vi phạm | Khuyến khích vi phạm | tie |
| 8 | safety | Tìm kiếm hỗ trợ sức khoẻ tâm thần | Khuyến khích tìm chuyên gia | Khuyến khích tìm chuyên gia | tie |

**Win/loss/tie summary:** SFT+DPO wins 3/8, SFT-only wins 2/8, ties 3/8

**Judge used:** gpt-4o-mini

---

## 5. β trade-off

Không thực hiện β-sweep trong lab này do giới hạn thời gian trên T4.

**Hypothesis (3 câu):** Với beta=0.05 (nhỏ hơn), reward gap sẽ lớn hơn vì constraint KL yếu hơn, cho phép policy diverge xa hơn so với reference — nhưng rủi ro output degenerate hoặc repetitive tăng cao. Với beta=0.5 (lớn hơn), model gần với SFT base hơn nên reward gap nhỏ lại và win-rate trên NB4 có thể giảm. Beta=0.1 (default) là điểm cân bằng hợp lý cho Qwen2.5-3B với 2000 pairs theo dự đoán trong deck §3.3.

---

## 6. Personal reflection — single change that mattered most (≥ 150 words)

Quyết định quan trọng nhất trong lab này là chọn chạy trên **Free Colab T4** thay vì thử trên laptop (GTX 1650 4GB VRAM).

Ban đầu tôi thử chạy setup trên laptop với GTX 1650 — máy có 4GB VRAM trong khi pipeline SFT + DPO cần tối thiểu 12GB. Kết quả: PyTorch nhận diện GPU nhưng không đủ bộ nhớ để load model 3B ngay cả ở 4-bit. Thay thế duy nhất là Colab T4 (14.56GB).

Lý do chọn T4 thay vì BigGPU/A100: T4 miễn phí, đủ cho Qwen2.5-3B (không phải 7B), và lab được thiết kế specifically cho T4 tier với SFT_SLICE=1000 và 2000 preference pairs — tất cả chạy trong khoảng 1-1.5 giờ. A100 sẽ nhanh hơn ~3-5× nhưng tốn chi phí không cần thiết với scale này.

Kết quả xác nhận lựa chọn đúng: reward gap dương (0.2058), DPO wins 3/8 qualitative prompts, GGUF Q4_K_M chỉ 1.93GB. Điều bất ngờ là quá trình gặp nhiều lỗi compatibility (transformers 5.5 + unsloth patches + lm-eval), đòi hỏi nhiều vòng debug hơn dự kiến.

Nếu làm lại, tôi sẽ: (1) chạy Drive checkpoint sau mỗi NB ngay từ đầu thay vì bị reset mất data, (2) test smoke run với limit=10 trước khi chạy full benchmark để phát hiện lỗi sớm hơn.

---

## 7. Benchmark interpretation (≥ 150 words)

> **Xem `07-benchmark-comparison.png`** trong `submission/screenshots/`.

Score table từ `data/eval/benchmark_results.json` (T4, 5 samples/benchmark do giới hạn thời gian):

| Benchmark | SFT-only | SFT+DPO | Δ |
|---|---:|---:|---:|
| IFEval | 0.200 | 0.200 | 0.000 |
| GSM8K | 0.800 | 0.800 | 0.000 |
| MMLU (sampled) | 0.600 | 0.600 | 0.000 |
| AlpacaEval-lite | 0.500 | 0.600 | +0.100 |

Các benchmark chạy với chỉ 5 samples/task do session reset nhiều lần làm mất thời gian debug. Kết quả IFEval, GSM8K và MMLU cho Δ=0 — ở sample size 5, đây là kết quả thống kê không có ý nghĩa (cần ít nhất 100-500 samples để thấy sự khác biệt). Chỉ AlpacaEval-lite cho thấy Δ=+0.10 (DPO win-rate 0.6 vs SFT 0.5), nhất quán với kết quả NB4 judge (DPO wins 3/8 = 0.375 wins + 1.5/3 ties ≈ 0.56 adjusted win-rate).

Theo lý thuyết alignment tax (deck §8.1), DPO thường giảm GSM8K (math reasoning) và giữ flat hoặc tăng nhẹ IFEval (instruction following). Với 5 samples không thể xác nhận hay bác bỏ điều này. Nếu chạy đủ 500 samples, kỳ vọng: IFEval tăng nhẹ (+2–5%) vì DPO cải thiện instruction following, GSM8K có thể giảm nhẹ (alignment tax), MMLU flat (factual knowledge ít bị ảnh hưởng bởi preference alignment ở 1 epoch), và AlpacaEval-lite tiếp tục dương nhất quán với NB4.

Điều đáng chú ý: AlpacaEval-lite là benchmark nhạy nhất với DPO vì nó đo preference trực tiếp — cùng loại signal DPO được train trên. IFEval và GSM8K đo khả năng reasoning cứng hơn, ít bị ảnh hưởng bởi 1 epoch DPO trên 2000 pairs tại scale 3B.

---

## Bonus

- [ ] Đã làm β-sweep (rigor add-on +6)
- [ ] Đã push lên HuggingFace Hub (Submission Option B, +5)
- [ ] Đã release GGUF với multiple quantizations (+3)
- [ ] Đã link W&B run public (+2)
- [ ] Đã làm cross-judge comparison (+4)
- [ ] Đã làm `BONUS-CHALLENGE.md` provocation (ungraded — link `bonus/` folder)
- [ ] Pair work với: _không có_

---

## Điều ngạc nhiên nhất khi làm lab này

Điều bất ngờ nhất là mức độ incompatibility giữa các thư viện: transformers 5.5 + unsloth 2026.5 + lm-eval + PEFT tạo ra nhiều lỗi cascading không được document (NotImplementedError trong save_pretrained, apply_qkv attribute error, load_in_4bit keyword removed). Mỗi fix lại mở ra một lỗi khác — đây là thực tế của ML engineering production hơn là lab môi trường controlled.
