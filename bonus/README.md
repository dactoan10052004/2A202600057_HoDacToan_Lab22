# Bonus: Tư Vấn Tuyển Sinh Đại Học — Vietnamese University Admissions Advisor

> Provocation tự tạo, lấy cảm hứng từ Provocation 1 (subject tutor) + Provocation 4 (domain-safe assistant).

**Contributors:** Ho Dac Toan (solo)

---

## Audience

Học sinh lớp 12 Việt Nam đang trong mùa xét tuyển đại học (cao điểm tháng 7–9 hằng năm). Đây là nhóm ~900.000 thí sinh/năm với đặc trưng:

- Lo lắng cao độ về điểm chuẩn, ngành học, trường phù hợp
- Nhận thông tin từ nhiều nguồn không chính thức (Facebook, TikTok, fanpage "chia sẻ kinh nghiệm")
- Dễ bị confused bởi thông tin cũ, sai năm, sai tổ hợp môn
- Không có thời gian đọc văn bản hành chính dài

**Secondary audience:** Phụ huynh hỗ trợ con em đăng ký.

---

## Domain Knowledge

Hệ thống tuyển sinh Việt Nam có đặc thù riêng mà ChatGPT generic không biết:

- **Tổ hợp môn:** A00 (Toán-Lý-Hoá), A01 (Toán-Lý-Anh), D01 (Toán-Văn-Anh), C00 (Văn-Sử-Địa)... Mỗi ngành chỉ xét một số tổ hợp nhất định.
- **Nguyện vọng:** Thí sinh được đăng ký tối đa 10 NV, sắp xếp theo thứ tự ưu tiên — hệ thống xét từ NV1 xuống.
- **Điểm chuẩn biến động:** Không dự đoán được chính xác, thay đổi mỗi năm theo số lượng thí sinh và chỉ tiêu từng trường.
- **Phương thức xét tuyển:** Thi THPTQG, xét học bạ, đánh giá tư duy (ĐGNL ĐHQG), chứng chỉ quốc tế (SAT, IELTS).
- **Nguồn chính thức:** Cổng thông tin tuyển sinh MOET (`tuyensinh.moet.gov.vn`), trang tuyển sinh từng trường, hotline 1800-XXXX của Bộ GD.
- **Ngành hot/cold theo thị trường VN:** CNTT, Kinh tế số, Y dược top điểm; Sư phạm điểm thấp hơn nhưng có chính sách học phí riêng.

Domain judgment cốt lõi: phân biệt được "thông tin nền hữu ích" vs "advice cá nhân hoá mà chỉ chuyên gia tư vấn trường mới đủ thông tin để đưa ra."

---

## Application Objective

Model hoạt động theo 3 nguyên tắc cứng:

1. **Inform, không Decide** — cung cấp thông tin nền + trỏ nguồn chính thức, không dự đoán "em sẽ đỗ/trượt"
2. **No hallucination về số liệu** — khi không có điểm chuẩn chính xác, nói rõ "cần kiểm tra trên tuyensinh.moet.gov.vn" thay vì bịa số
3. **Empathetic, không lạnh lùng** — nhận ra học sinh đang stress, tone hỗ trợ, không bureaucratic

**DPO signal:**
- `chosen` = thông tin nền rõ + trỏ nguồn chính thức + khuyến khích verify + tone ấm
- `rejected` = dự đoán điểm chuẩn cụ thể (hallucination risk) HOẶC từ chối lạnh "tôi không biết" HOẶC trả lời generic bằng English mixed VN

---

## Real-World Output

| Deliverable | Mô tả | Status |
|---|---|---|
| `data/prompts.jsonl` | 200 prompts từ câu hỏi thực tế học sinh | planned |
| `data/pairs.parquet` | 200 preference pairs (chosen/rejected) | planned |
| `train.py` | DPO training script (reuse NB3 pipeline) | planned |
| `adapters/dpo-bonus/` | Fine-tuned adapter | planned |
| `demo/serve.py` | Gradio demo ~60 dòng | planned |
| `demo/5-samples.md` | 5 prompt + output before/after DPO | planned |
| `MODEL-CARD.md` | Model card với explicit "NOT for" list | planned |

**Target:** Ai clone repo + `python bonus/demo/serve.py` → có Gradio UI tương tác được trong 2 phút.

---

## What This Model Is NOT For

- Dự đoán điểm chuẩn cho năm tới
- Tư vấn cá nhân hoá ("với học bạ của em thì nên đăng ký đâu")
- Thay thế tư vấn tuyển sinh chính thức từ trường
- Thông tin tuyển sinh sau đại học, học bổng, du học

---

## 5 Sample Interactions (Planned)

```
1. "Tổ hợp A01 thì thi được những ngành nào ở BKHN?"
   → Giải thích A01 = Toán-Lý-Anh, list ngành phù hợp, link tuyensinh.hust.edu.vn

2. "Em 25 điểm thi THPT, đăng ký CNTT trường nào được?"
   → KHÔNG dự đoán đỗ/trượt, giải thích cơ chế điểm chuẩn biến động,
     gợi ý cách tra cứu điểm chuẩn năm trước, link MOET

3. "Học bạ của em toàn 8.5 thì xét học bạ có vào được Ngoại thương không?"
   → Giải thích phương thức xét học bạ của FTU, NOT dự đoán, khuyến khích
     liên hệ phòng tuyển sinh FTU trực tiếp

4. "Ngành Kế toán hay QTKD học xong dễ kiếm việc hơn?"
   → Thông tin thị trường lao động VN, tránh khẳng định tuyệt đối,
     gợi ý check thêm báo cáo lao động Bộ LĐTBXH

5. "Em muốn học Y mà bố mẹ bắt học Kinh tế, làm sao?"
   → Out of scope (tư vấn gia đình), acknowledge cảm xúc, gợi ý
     nói chuyện với thầy cô chủ nhiệm/chuyên viên tư vấn tâm lý
```

---

## Honest Limitations

- **Data staleness:** Điểm chuẩn thay đổi hằng năm — model không có real-time data, luôn phải refer ra nguồn chính thức
- **Scale:** 200 pairs / 1 epoch là POC, không phải production
- **Bias:** Data xây từ perspective học sinh miền Bắc (người build), có thể thiếu context trường miền Nam/Trung
- **Privacy:** Không lưu thông tin điểm số cá nhân của người dùng
- **License:** Qwen2.5 base — không dùng cho commercial deployment không có attribution
