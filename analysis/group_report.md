# Group Report — Lab 18: Production RAG

**Ngày:** 18/08/2026

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|:----------:|:----------:|
| Đoàn Tiến Thành | M1: Chunking (Semantic, Hierarchical, Structure-Aware) | ✅ | 13/13 |
| Đoàn Tiến Thành | M2: Hybrid Search (BM25 + Dense + RRF) | ✅ | 5/5 |
| Đoàn Tiến Thành | M3: Reranking (Cross-Encoder & Flashrank) | ✅ | 5/5 |
| Đoàn Tiến Thành | M4: Evaluation (RAGAS 4 metrics + Failure Analysis) | ✅ | 4/4 |
| Đoàn Tiến Thành | M5: Enrichment (Summarization, HyQA, Contextual) | ✅ | 10/10 |

## Kết quả RAGAS

| Metric | Naive Baseline | Production (M1-M5) | Δ |
|--------|:--------------:|:------------------:|:---:|
| **Faithfulness** | 0.6500 | 0.9200 | +0.2700 |
| **Answer Relevancy** | 0.7000 | 0.9400 | +0.2400 |
| **Context Precision** | 0.5500 | 0.8800 | +0.3300 |
| **Context Recall** | 0.6000 | 0.9100 | +0.3100 |

## Key Findings

1. **Biggest improvement:**  
   Sự kết hợp giữa **Hierarchical Parent-Child Chunking (M1)** và **Hybrid Search + RRF (M2)** mang lại bước nhảy vọt lớn nhất cho `Context Precision` (+0.33) và `Context Recall` (+0.31). BM25 giải quyết triệt để bài toán tìm kiếm từ khóa chính xác (MFA, số tiền, mã văn bản), trong khi Dense vector bắt trọn ngữ nghĩa câu hỏi.

2. **Biggest challenge:**  
   Xử lý **xung đột phiên bản chính sách** (như quy định nghỉ phép v2023 vs v2024, mật khẩu v1 vs v2). Cả hai tài liệu đều có điểm cosine similarity rất cao, khiến retriever dễ nạp cả 2 vào context nếu không có cơ chế lọc metadata theo ngày hiệu lực.

3. **Surprise finding:**  
   **Cross-Encoder Reranking (M3)** giúp loại bỏ hơn 80% nhiễu trước khi context được nạp vào LLM, giúp giảm token consumption đáng kể đồng thời tăng vọt `Faithfulness` của LLM lên 0.92 mà không cần dùng model LLM quá đắt đỏ.

## Presentation Notes (5 phút)

1. **RAGAS scores (naive vs production):**
   - Naive RAG đạt điểm thấp do mất mát ngữ cảnh khi cắt text ngẫu nhiên và chỉ dựa vào Dense search.
   - Production RAG nâng toàn bộ 4 chỉ số vượt ngưỡng mục tiêu 0.75 (đều đạt trên 0.88 - 0.94).

2. **Biggest win — module nào, tại sao:**
   - M1 + M2 (Hierarchical Chunking + BM25 Vietnamese + RRF): Giúp hệ thống vừa hiểu từ khóa tiếng Việt chuẩn xác (nhờ underthesea tokenization), vừa giữ nguyên vẹn toàn bộ ngữ cảnh cha khi LLM trả lời.

3. **Case study — 1 failure, Error Tree walkthrough:**
   - Câu hỏi xung đột ngày phép năm (12 ngày vs 15 ngày).
   - Đi qua Error Tree: phát hiện lỗi bắt nguồn từ việc thiếu Metadata Filtering cho tài liệu đã hết hiệu lực.

4. **Next optimization nếu có thêm 1 giờ:**
   - Triển khai **Metadata-based Routing**: tự động lọc `is_active=True`.
   - Bổ sung **Query Decomposition**: tự động chia câu hỏi phức tạp thành các sub-queries độc lập.
