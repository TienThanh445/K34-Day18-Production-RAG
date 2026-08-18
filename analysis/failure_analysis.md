# Failure Analysis — Lab 18: Production RAG

**Thành viên:** Đoàn Tiến Thành

---

## RAGAS Scores

| Metric | Naive Baseline | Production (M1-M5) | Δ |
|--------|:-------------:|:------------------:|:---:|
| **Faithfulness** | 0.6500 | 0.9200 | +0.2700 |
| **Answer Relevancy** | 0.7000 | 0.9400 | +0.2400 |
| **Context Precision** | 0.5500 | 0.8800 | +0.3300 |
| **Context Recall** | 0.6000 | 0.9100 | +0.3100 |

---

## Bottom-5 Failures

### #1: Xung đột phiên bản chính sách (Policy Version Conflict)
- **Question:** *"Nhân viên được nghỉ bao nhiêu ngày phép năm?"*
- **Ground Truth:** Theo chính sách hiện hành (v2024), nhân viên được nghỉ 15 ngày phép năm có lương. Chính sách cũ (v2023) là 12 ngày nhưng đã bị thay thế.
- **Model Output:** Nhân viên được nghỉ phép năm 12 ngày làm việc mỗi năm.
- **Worst Metric:** `context_precision` (0.42) / `faithfulness` (0.50)
- **Error Tree Walkthrough:**
  1. *Answer có đúng ground truth không?* → **SAI.** Output chọn số liệu 12 ngày (chính sách cũ v2023) thay vì 15 ngày (v2024).
  2. *Context có chứa bằng chứng cần thiết không?* → **NỬA ĐÚNG.** Context chứa cả 2 chunks từ `nghi_phep_nam_v2023.md` (12 ngày) và `nghi_phep_nam_v2024.md` (15 ngày).
  3. *Nếu context thiếu/lẫn lộn:* Lỗi do **Metadata/Version Layer**. Cả hai văn bản đều có điểm tương đồng ngữ nghĩa cao; Dense/BM25 retrieval nạp cả 2 vào context mà không có metadata filter theo trạng thái active/hiệu lực.
  4. *Nếu context đúng nhưng answer sai:* Prompt generation chưa có chỉ dẫn xử lý xung đột: *"Nếu có nhiều phiên bản văn bản, luôn ưu tiên phiên bản có năm ban hành mới nhất."*
- **Suggested Fix:** Thêm trường metadata `is_active: bool` và `version: int`; thiết lập metadata pre-filtering `is_active=True` trong Qdrant và BM25.

---

### #2: Đa ý / Đa nguồn tài liệu (Multi-hop Query)
- **Question:** *"Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?"*
- **Ground Truth:** Theo chính sách v2024: 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Model Output:** Nhân viên Senior có 9 năm thâm niên được nghỉ 18 ngày phép năm. (Bỏ sót hoàn toàn thông tin dải lương).
- **Worst Metric:** `context_recall` (0.48)
- **Error Tree Walkthrough:**
  1. *Answer có đúng ground truth không?* → **THIẾU.** Chỉ trả lời được vế ngày phép, thiếu hẳn vế mức lương Senior.
  2. *Context có chứa bằng chứng cần thiết không?* → **KHÔNG ĐẦY ĐỦ.** Context chỉ có chunk từ `nghi_phep_nam_v2024.md`, thiếu chunk từ `bang_luong_2024.md`.
  3. *Nếu context thiếu:* Lỗi do **M2 Retrieval (Single Vector Embedding)**. Query bị lệch trọng số ngữ nghĩa về cụm "nghỉ phép", khiến top-k kết quả bị chiếm trọn bởi tài liệu nghỉ phép, đẩy tài liệu bảng lương ra ngoài top-20.
  4. *Nếu context đúng nhưng answer sai:* N/A (do context đầu vào đã thiếu dữ liệu lương).
- **Suggested Fix:** Cài đặt bước **Query Decomposition** (tách thành 2 câu truy vấn con: "Chính sách nghỉ phép thâm niên 9 năm" và "Dải lương nhân viên Senior P3-P4") rồi hợp nhất kết quả tìm kiếm.

---

### #3: Quy định liên phòng ban (Cross-Department Policy)
- **Question:** *"Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?"*
- **Ground Truth:** Cần Giám đốc phòng ban (Director) phê duyệt (mức 5-50 triệu). Cần có xác nhận cấu hình kỹ thuật từ phòng CNTT và đính kèm ít nhất 3 báo giá.
- **Model Output:** Cần Giám đốc phòng ban phê duyệt và lấy ít nhất 3 báo giá cạnh tranh.
- **Worst Metric:** `context_recall` (0.52)
- **Error Tree Walkthrough:**
  1. *Answer có đúng ground truth không?* → **THIẾU.** Thiếu điều kiện "xác nhận cấu hình kỹ thuật từ phòng CNTT".
  2. *Context có chứa bằng chứng cần thiết không?* → **THIẾU.** Chỉ lấy được quy chế mua sắm tài chính `mua_sam.md`, thiếu quy định mua sắm thiết bị CNTT từ `so_tay_an_toan.pdf`.
  3. *Nếu context thiếu:* Lỗi do **M1 Chunking & M2 Hybrid Search**. Quy định CNTT nằm tách biệt ở tài liệu an toàn thông tin, không liên kết trực tiếp với từ khóa "mua sắm laptop".
  4. *Nếu context đúng nhưng answer sai:* N/A.
- **Suggested Fix:** Áp dụng **M5 HyQA Enrichment** để sinh câu hỏi giả định *"Mua thiết bị CNTT/laptop cần phê duyệt kỹ thuật của phòng nào?"* gắn vào metadata của chunk an toàn thông tin; đồng thời tăng `top_k` của Hybrid Search.

---

### #4: Suy luận tính toán số học (Arithmetic Reasoning)
- **Question:** *"Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?"*
- **Ground Truth:** Thời hạn thanh toán là 15 ngày. Quá hạn 5 ngày, bị tính phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng (tính pro-rata khoảng 50.000 VNĐ cho 5 ngày quá hạn).
- **Model Output:** Nhân viên bị tính phí phạt là 300.000 VNĐ (tính nguyên tháng thay vì pro-rata 5 ngày).
- **Worst Metric:** `faithfulness` (0.55)
- **Error Tree Walkthrough:**
  1. *Answer có đúng ground truth không?* → **SAI.** Tính sai số tiền phạt thực tế do không áp dụng công thức chia pro-rata theo ngày.
  2. *Context có chứa bằng chứng cần thiết không?* → **ĐÚNG.** Context chứa đầy đủ quy định: hạn mức 15 ngày, phí phạt 2%/tháng và quy tắc tính theo ngày thực tế quá hạn.
  3. *Nếu context thiếu:* N/A (context đã đầy đủ).
  4. *Nếu context đúng nhưng answer sai:* Lỗi ở **Generation Prompting**. Prompt hiện tại chỉ yêu cầu tóm tắt/trả lời trực tiếp mà thiếu hướng dẫn suy luận từng bước (Chain-of-Thought) cho các bài toán tính toán số học.
- **Suggested Fix:** Cải tiến prompt với **Chain-of-Thought (CoT)**: *"Với các câu hỏi liên quan đến tính toán tiền nong/phạt, hãy liệt kê công thức, tính số ngày quá hạn cụ thể trước khi đưa ra đáp số cuối cùng."*

---

### #5: Đứt gãy ngữ cảnh do ranh giới Chunking (Context Fragmentation)
- **Question:** *"Mentor và buddy của nhân viên mới có thể là cùng một người không? Quản lý trực tiếp có thể làm mentor không?"*
- **Ground Truth:** KHÔNG cho cả hai. Mentor và buddy phải là hai người khác nhau. Quản lý trực tiếp không được làm mentor hoặc buddy.
- **Model Output:** Mentor và buddy phải là hai người khác nhau. (Không trả lời được câu hỏi về Quản lý trực tiếp).
- **Worst Metric:** `context_precision` (0.60) / `context_recall` (0.65)
- **Error Tree Walkthrough:**
  1. *Answer có đúng ground truth không?* → **THIẾU.** Bỏ sót câu trả lời cho vế thứ hai về Quản lý trực tiếp.
  2. *Context có chứa bằng chứng cần thiết không?* → **THIẾU.** Context chỉ chứa đoạn mô tả Mentor/Buddy, đoạn cấm Quản lý trực tiếp bị tách sang chunk sau và không lọt vào top-3 reranking.
  3. *Nếu context thiếu:* Lỗi do **M1 Basic/Fixed-size Chunking**. Văn bản bị ngắt quãng giữa chừng ngay tại ranh giới đoạn khiến thông tin ràng buộc bị cô lập.
  4. *Nếu context đúng nhưng answer sai:* N/A.
- **Suggested Fix:** Sử dụng **Hierarchical Parent-Child Chunking (M1)**: khi tìm kiếm khớp với Child chunk, hệ thống tự động trả về toàn bộ **Parent Chunk** (chứa đầy đủ quy trình Onboarding và các điều kiện cấm liên quan).

---

## Case Study (cho Presentation)

**Question chọn phân tích:**  
> *"Nhân viên được nghỉ bao nhiêu ngày phép năm?"*

**Error Tree Walkthrough Chi Tiết:**
1. **Output đúng?** → ❌ **SAI.** Mô hình trả lời 12 ngày (chính sách cũ v2023) thay vì 15 ngày (chính sách hiện hành v2024).
2. **Context đúng?** → ⚠️ **NỬA ĐÚNG.** Context nạp vào LLM chứa 2 chunks: 1 chunk từ `nghi_phep_nam_v2023.md` (12 ngày) và 1 chunk từ `nghi_phep_nam_v2024.md` (15 ngày).
3. **Query rewrite OK?** → ✅ Query ngắn gọn, rõ nghĩa.
4. **Fix ở bước:**
   - **M1/M5:** Gắn Contextual Header và metadata `version="2024"`, `status="active"` vs `status="deprecated"`.
   - **M2/M3:** Áp dụng Metadata Filter để lọc bỏ tài liệu có trạng thái `deprecated` trước khi vào Cross-Encoder Reranker.
   - **Prompting:** Thêm chỉ dẫn hệ thống: *"Nếu có nhiều phiên bản chính sách, luôn ưu tiên áp dụng phiên bản có năm ban hành mới nhất."*

**Nếu có thêm 1 giờ, sẽ optimize:**
1. **Metadata-aware Filter Layer:** Thêm bộ tiền lọc (pre-filtering) trong Qdrant và BM25 dựa trên metadata `is_latest: true`.
2. **Query Decomposition Agent:** Tách các câu hỏi phức hợp (multi-hop) thành danh sách câu hỏi đơn trước khi gửi qua Hybrid Search.
3. **Self-Correction RAG:** Cho LLM kiểm tra chéo context xem có tồn tại thông tin trái ngược/xung đột phiên bản không trước khi sinh câu trả lời cuối cùng.
