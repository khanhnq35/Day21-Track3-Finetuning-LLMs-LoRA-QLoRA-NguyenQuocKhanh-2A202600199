# Lab 21 — Evaluation Report

**Học viên**: Nguyễn Quốc Khánh — khanhnq3505
**Ngày nộp**: 2026-05-07
**Submission option**: Option B (HF Hub)

## 1. Setup
- **Base model**: `unsloth/Llama-3.2-3B-Instruct-bnb-4bit`
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, 500 samples (450 train + 50 eval)
- **max_seq_length**: 1024 (p95 = 562)
- **Target Modules**: ALL LAYERS (`["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`)
- **GPU**: Tesla T4, 16 GB VRAM
- **Training cost**: ~$0.15 (~35 phút tổng cộng cho cả 3 rank @ $0.2/hr)
- **HF Hub links**: 
  - Rank 8: https://huggingface.co/khanhnq3505/Llama-3.2-3B-Vietnamese-Lab21-r8
  - Rank 16: https://huggingface.co/khanhnq3505/Llama-3.2-3B-Vietnamese-Lab21-r16
  - Rank 64: https://huggingface.co/khanhnq3505/Llama-3.2-3B-Vietnamese-Lab21-r64

## 2. Rank Experiment Results

| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-----------------|------------|-----------|-----------|------------|
| 8    | 12,156,928      | 10.45 min  | 7.15 GB   | 1.5465    | 4.69       |
| 16   | 24,313,856      | 10.54 min  | 6.47 GB   | 1.5700    | 4.80       |
| 64   | 97,255,424      | 11.05 min  | 8.80 GB   | 1.6803    | 5.36       |

*(Lưu ý: Do áp dụng target ALL layers nên số lượng tham số cao hơn bình thường)*

## 3. Loss Curve Analysis
- **Quan sát**: Đường Training Loss có xu hướng giảm rõ rệt và ổn định dần sau 3 epoch (từ ~1.61 xuống ~1.39).
- **Đánh giá Overfitting**: Ở Rank cơ sở (như r=16), Eval Loss (~1.51) nhỉnh hơn một chút so với Train Loss (~1.39). Khoảng cách gap này khá nhỏ (~0.12), cho thấy model không bị overfitting trong quá trình train từng rank cụ thể, mà đang có khả năng tổng quát hóa tốt. Có xuất hiện nhiễu (noise) trên biểu đồ do batch size nhỏ.

## 4. Qualitative Comparison (5 examples)

### Example 1
**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.
- **Base**: Trả lời đúng nhưng bị lặp từ và ngắt quãng ("thực hiện các tác...").
- **Fine-tuned**: Giải thích chuyên nghiệp hơn, nhấn mạnh việc "không có sự hướng dẫn trực tiếp từ người dùng".
- **Nhận xét**: Fine-tuned cải thiện rõ rệt về cấu trúc và văn phong tiếng Việt tự nhiên hơn.

### Example 2
**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.
- **Base**: Code trả về 0 cho n=1, 1 cho n=2 (khác với chuẩn thông thường).
- **Fine-tuned**: Thêm phần xử lý lỗi `raise ValueError` cho input âm. Dùng vòng lặp `a, b = 0, 1` tối ưu và chuẩn xác hơn.
- **Nhận xét**: Improved. Model fine-tuned có tư duy lập trình chặt chẽ và an toàn hơn.

### Example 3
**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.
- **Base**: Câu văn lủng củng, lặp từ ("thân thiện... thân thiện").
- **Fine-tuned**: Liệt kê rành mạch 4 nguyên tắc (Chuyển đổi, Thích ứng, Đơn giản, Tương thích).
- **Nhận xét**: Improved. Khả năng trình bày danh sách (listing) tốt hơn hẳn.

### Example 4
**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.
- **Base**: Trả lời vòng vo, lặp ý ("cải thiện hiệu năng... bằng cách...").
- **Fine-tuned**: Cố gắng giải thích chi tiết hơn nhưng bị "hallucinate" (bịa) cụm từ viết tắt của LoRA thành Layer-wise Adaptive...
- **Nhận xét**: Degraded/Mixed. Model fine-tuned có văn phong tốt hơn nhưng bịa kiến thức (hallucination) ở thuật ngữ chuyên ngành.

### Example 5
**Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.
- **Base**: Giải thích dài dòng, câu văn bị ngắt giữa chừng.
- **Fine-tuned**: Đi thẳng vào vấn đề, giải thích gọn gàng từng khái niệm.
- **Nhận xét**: Improved. Khả năng tóm tắt và tập trung vào trọng tâm tốt hơn.

## 5. Conclusion về Rank Trade-off

Qua thực nghiệm với việc target **tất cả các lớp (All Layers)**, em nhận thấy một hiện tượng đặc biệt: **Rank 8 lại mang lại chỉ số Perplexity tốt nhất (4.69)**. Khi tăng Rank lên 16 và 64, Perplexity không những không giảm mà lại tăng lên (4.80 và 5.36). 

Điều này xảy ra là do hiện tượng **Overfitting** khi tăng sức chứa của model. Vì em chọn target All Layers, ngay cả ở Rank 8, số lượng tham số học được đã lên tới 12 triệu. Với một dataset nhỏ (chỉ 500 mẫu), việc tăng Rank lên 64 (gần 100 triệu tham số) khiến model bị "bão hòa" và học vẹt dữ liệu train, dẫn đến kém hiệu quả trên tập đánh giá. 

**Recommendation**: Nếu deploy production cho bài toán này với dataset hiện tại, em sẽ chọn **Rank 8**. Rank 8 cho ROI tốt nhất: tiết kiệm VRAM, thời gian train nhanh nhất và quan trọng là độ chính xác ngôn ngữ (Perplexity) tốt nhất vì không bị Overfitting do dư thừa tham số.

## 6. What I Learned
- Hiểu được sức mạnh của việc nhắm mục tiêu (target) All Layers: Mặc dù tốn thêm tài nguyên nhưng khả năng logic và suy luận của model được cải thiện rất mạnh (đặc biệt trong task code).
- Nắm được bài học thực tế về "Diminishing Returns" và Overfitting: Không phải cứ Rank cao là tốt, đặc biệt khi dataset có kích thước nhỏ. Việc giám sát Perplexity là cực kỳ quan trọng để chọn điểm dừng.
- Nắm vững quy trình làm việc chuyên nghiệp: Format dataset chuẩn, tối ưu max_seq_length, và push adapter lên HuggingFace Hub để dễ dàng tích hợp và chia sẻ.
