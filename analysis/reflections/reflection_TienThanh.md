# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Đoàn Tiến Thành  

---

## Phần 1: Mapping bài giảng (10 phút)

Dưới đây là liên kết trực tiếp giữa các khái niệm lý thuyết cốt lõi trong bài giảng Production RAG và mã nguồn thực tế đã cài đặt:

| Lecture Concept | Module | Hàm / Class cụ thể | Observation & Ghi chú thực nghiệm |
|:---|:---:|:---|:---|
| **Semantic Chunking** | M1 | `chunk_semantic()` | Dùng cosine similarity giữa các vector câu (`all-MiniLM-L6-v2`) với ngưỡng `threshold=0.85`. Cho phép nhóm các câu liền kề cùng chủ đề tự nhiên, tránh bị cắt đứt giữa câu so với paragraph splitting cố định. |
| **Hierarchical Chunking (Parent-Child)** | M1 | `chunk_hierarchical()` | Tách parent lớn (`2048` chars) để làm context cho LLM và child nhỏ (`256` chars) để tăng độ chính xác của vector search. Child luôn mang `parent_id` liên kết chặt chẽ với Parent. |
| **Structure-Aware Chunking** | M1 | `chunk_structure_aware()` | Parse Markdown header cấp 1–3 (`#`, `##`, `###`), giữ nguyên khối bảng biểu, danh sách và code block không bị xé vụn giữa chừng; tự động lưu tiêu đề mục vào `metadata["section"]`. |
| **Vietnamese Word Tokenization** | M2 | `segment_vietnamese()` | Dùng `underthesea.word_tokenize(..., format="text")` và thay thế `_` thành khoảng trắng ` ` để BM25 tokenization khớp đúng từ ghép tiếng Việt với câu truy vấn người dùng. |
| **Dense Vector Indexing** | M2 | `DenseSearch.index()`, `search()` | Quản lý vector database với Qdrant Client, cấu hình vector dimension `1024` (`BAAI/bge-m3`), lưu toàn bộ payload metadata và query vector bằng API `query_points()`. |
| **Reciprocal Rank Fusion (RRF)** | M2 | `reciprocal_rank_fusion()` | Hợp nhất thứ hạng độc lập giữa BM25 (từ khóa chính xác) và Dense Search (ngữ nghĩa tương đồng) theo công thức $\sum \frac{1}{k + rank}$. Giúp tài liệu xuất hiện đồng thời ở cả 2 list vươn lên vị trí số 1. |
| **Cross-Encoder Re-ranking** | M3 | `CrossEncoderReranker.rerank()` | Rerank top-20 ứng viên xuống top-3 bằng Cross-Encoder (`BAAI/bge-reranker-v2-m3`). Đọc toàn bộ cặp `(query, document)` để chấm điểm tương tác sâu, lọc bỏ trên 80% văn bản gây nhiễu context. |
| **RAGAS 4 Core Metrics** | M4 | `evaluate_ragas()` | Đánh giá khách quan 4 khía cạnh: `Faithfulness` (tránh ảo giác), `Answer Relevancy` (trúng đích câu hỏi), `Context Precision` (độ tập trung context), `Context Recall` (độ đầy đủ context). |
| **Diagnostic Tree Failure Analysis** | M4 | `failure_analysis()` | Tự động phân tích các câu hỏi điểm thấp nhất, tìm ra `worst_metric` và ánh xạ sang nguyên nhân cốt lõi (Retrieval, Chunking, hay LLM Generation) để có hành động khắc phục tương ứng. |
| **Contextual Prepend (Anthropic style)** | M5 | `contextual_prepend()` | Gắn ngữ cảnh vị trí tài liệu vào đầu mỗi chunk trước khi embed, giải quyết triệt để tình trạng chunk bị cô lập ngữ cảnh khi đứng độc lập. |
| **HyQA & Single-call Enrichment** | M5 | `_enrich_single_call()` | Gộp 4 tác vụ (Tóm tắt, Sinh câu hỏi giả định, Prepend context, Trích xuất metadata) vào **1 LLM call duy nhất**, tối ưu hóa 75% chi phí API so với việc gọi riêng lẻ. |

---

## Phần 2: Khó khăn & Cách giải quyết (10 phút)

### 1. Sự cố cảnh báo NumPy trên Python 3.13 Windows
- **Exact Error / Warning:**
  ```
  <frozen importlib._bootstrap>:488: Warning: Numpy built with MINGW-W64 on Windows 64 bits is experimental...
  D:\...\numpy\core\getlimits.py:225: RuntimeWarning: invalid value encountered in exp2
  ```
- **Cách debug:** Kiểm tra môi trường thực thi thấy Python phiên bản 3.13 đi kèm với bản build cũ `numpy 1.26.4` được biên dịch thử nghiệm bằng MinGW-w64.
- **Cách giải quyết:** Nâng cấp `numpy` lên phiên bản `2.x` (`pip install --upgrade numpy`) được Microsoft Visual C++ (MSVC) hỗ trợ chính thức trên Windows, loại bỏ triệt để cảnh báo và lỗi số học.

### 2. Xung đột Tokenizer tiếng Việt trong BM25
- **Hiện tượng:** Truy vấn từ khóa chính xác tiếng Việt bằng BM25 ban đầu bị điểm 0 hoặc match sai tài liệu.
- **Cách debug:** `underthesea` mặc định nối từ ghép bằng dấu gạch dưới (ví dụ `nghỉ_phép`), trong khi BM25 tách từ bằng khoảng trắng. Câu truy vấn `nghỉ phép` bị tách thành 2 tokens rời, không khớp với token `nghỉ_phép`.
- **Cách giải quyết:** Chuẩn hóa đầu ra trong hàm `segment_vietnamese()` bằng `.replace("_", " ")` trước khi đưa vào BM25 corpus tokens.

### 3. Vấn đề nghẽn mạng khi tải các mô hình lớn (Unauthenticated HF Download)
- **Hiện tượng:** Hugging Face áp dụng cơ chế bóp băng thông khi tải mô hình `BAAI/bge-m3` (~2GB) và `BAAI/bge-reranker-v2-m3` (~2.2GB) khi không có `HF_TOKEN`.
- **Cách giải quyết:** 
  - Cấu hình biến môi trường `HF_TOKEN` hoặc đăng nhập `huggingface-cli login`.
  - Thiết kế pipeline mô-đun hóa độc lập, xây dựng fallback logic cho unit test khi môi trường offline.

---

## Phần 3: Action Plan cho project (10 phút)

## Project: Hệ thống CSKH & Xử lý Khiếu nại Ví Điện Tử Thông Minh (E-Wallet Support & Dispute Resolution)

### Hiện tại
- **RAG pipeline hiện tại:** Đang sử dụng Naive RAG cơ bản (cắt paragraph cố định 500 ký tự, Dense search đơn thuần qua vector embeddings, chưa có bộ lọc từ khóa, không có reranker).
- **Known issues:**
  1. **Bắt mã lỗi và thuật ngữ nghiệp vụ kém:** Khách hàng tra cứu mã lỗi giao dịch (VD: `ERR_TRANS_901`, `NAPAS_TIMEOUT`, `REFUND_PENDING`) nhưng vector search chỉ bắt ngữ nghĩa chung chung, không tìm đúng quy trình xử lý mã lỗi cụ thể.
  2. **Xung đột quy chế hoàn tiền & SLA thời gian xử lý:** Quy chế hoàn tiền cho các kênh giao dịch khác nhau (Ví sang Ví: tức thì; Ngân hàng nội địa: 1–3 ngày làm việc; Thẻ quốc tế Visa/Mastercard: 7–14 ngày làm việc) bị lẫn lộn giữa phiên bản chính sách cũ và mới cập nhật.
  3. **Đứt gãy ngữ cảnh bảng biểu phí & hạn mức:** Các bảng biểu hạn mức giao dịch theo cấp độ xác thực (Chưa KYC, KYC Cấp 1, KYC Cấp 2) bị cắt vụn qua ranh giới chunking cố định khiến LLM tư vấn sai hạn mức chuyển/rút tiền của khách hàng.

### Kế hoạch áp dụng các kỹ thuật từ Lab 18

| Hạng mục | Kỹ thuật áp dụng | Mục tiêu cải thiện | Timeline |
|:---|:---|:---|:---:|
| **1. Chunking** | Áp dụng **Structure-Aware Chunking** cho quy chế dịch vụ & bảng mã lỗi; dùng **Hierarchical Parent-Child Chunking** cho quy trình xử lý khiếu nại (4 bước hoàn tiền). | Giữ nguyên vẹn bảng biểu hạn mức, bảng mã lỗi; tăng `Context Precision` lên > 0.88. | Tuần 1 |
| **2. Search Layer** | Kết hợp **BM25 tiếng Việt (underthesea)** + **Dense Vector (`bge-m3`)** với thuật toán **RRF (k=60)**. | Bắt chính xác 100% mã lỗi (`ERR_...`), tên ngân hàng đối tác, mã đối soát; tăng `Context Recall` lên > 0.92. | Tuần 2 |
| **3. Re-ranking** | Tích hợp **Cross-Encoder Reranker (`bge-reranker-v2-m3`)** rút gọn top-20 ứng viên xuống top-3 context chuẩn nhất. | Phân biệt chính xác tình huống nạp tiền vs chuyển tiền vs thanh toán hóa đơn; tăng `Faithfulness` lên > 0.95. | Tuần 2 |
| **4. Metadata Layer** | Xây dựng bộ lọc **Metadata Pre-filtering** theo `service_type` (`nap_tien`, `chuyen_tien`, `hoan_tien`), `channel` (`napas`, `visa`, `vi_vi`), `kyc_level` và `status: active`. | Loại bỏ hoàn toàn lỗi áp dụng nhầm chính sách hoàn tiền cũ hoặc kênh thanh toán không tương ứng. | Tuần 3 |
| **5. Evaluation CI/CD** | Xây dựng bộ test set 50 câu hỏi tình huống khiếu nại (chuyển nhầm tiền, giao dịch treo, nghi ngờ gian lận) đo lường qua **RAGAS 4 metrics** và Diagnostic Tree. | Giám sát chất lượng tự động, phát hiện sớm regression trước khi release production. | Tuần 3 |
