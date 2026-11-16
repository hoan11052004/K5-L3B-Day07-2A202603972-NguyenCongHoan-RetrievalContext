# Chi tiết Bộ Benchmark 24 Câu hỏi (Day 7 — Agentic RAG)

Bộ benchmark gồm **24 câu hỏi** được thiết kế chuẩn theo yêu cầu của [BENCHMARK.md](../BENCHMARK.md), chia làm 2 tập độc lập:
- **Tập Dev (17 câu):** 15 câu có đáp án  + 2 câu thiếu bằng chứng.
- **Tập Test (7 câu):** 5 câu có đáp án  + 2 câu thiếu bằng chứng.

---

## 1. Tập Dev (17 câu)

### A. 15 câu có đáp án (Answerable)

| ID | Câu hỏi | Đáp án kỳ vọng | Trang | Bằng chứng (Gold Evidence) |
|---|---|---|:---:|---|
| **`q01`** | Một thuật toán học máy được định nghĩa cơ bản là thuật toán có khả năng làm gì? | Có khả năng học từ dữ liệu | 3 | "Một thuật toán học máy là thuật toán có khả năng học từ dữ liệu." |
| **`q02`** | Các feature của một hình ảnh trong bài toán Machine Learning thường là gì? | Các giá trị độ sáng của các pixel | 4 | "các feature thường là các giá trị độ sáng của các pixel" |
| **`q03`** | Trong tác vụ tổng hợp và lấy mẫu, thuật toán ML được yêu cầu tạo ra đối tượng nào? | Tạo ra các mẫu mới tương tự như dữ liệu huấn luyện | 7 | "thuật toán học máy được yêu cầu sinh ra các mẫu mới tương tự như dữ liệu huấn luyện" |
| **`q04`** | Tập dữ liệu hoa Diên Vĩ (Iris) đo đạc các bộ phận của bao nhiêu cá thể cây? | 150 cá thể cây | 10 | "tập dữ liệu đo đạc những phần khác nhau của 150 cá thể cây thuộc các loại Diên Vĩ khác nhau." |
| **`q05`** | Trong các phép biến đổi tịnh tiến, hệ số chặn thường được gọi là gì? | Tham số điều chỉnh độ dốc và độ chệch (bias) | 16 | "phép biến đổi tịnh tiến thường được gọi là độ chệch (bias)" |
| **`q06`** | Theo định lý No Free Lunch, có thuật toán ML nào luôn luôn tốt hơn các thuật toán còn lại không? | Không có thuật toán ML nào luôn luôn tốt hơn hơn các thuật toán ML còn lại | 25 | "không có thuật toán ML nào luôn luôn tốt hơn hơn các thuật toán ML còn lại." |
| **`q07`** | Tập validation được sử dụng để làm gì đối với các siêu tham số trong quá trình huấn luyện? | Để học (hoặc lựa chọn/tinh chỉnh) các siêu tham số | 29 | "tập validation được sử dụng để chọn các siêu tham số" |
| **`q08`** | Điều kiện mô tả sự hội tụ theo xác suất trong phương trình 5.38 phát biểu như thế nào khi m tiến tới vô cùng? | P(\|θ̂_m - θ\| > ε) -> 0 khi m -> ∞ | 39 | "P(\|θ̂_m - θ\| > ε) -> 0 khi m -> ∞" |
| **`q09`** | Khi ước lượng một giá trị đơn lẻ của tham số, phương pháp tính toán này được gọi là gì? | Ước lượng điểm (Point estimation) | 44 | "Ước lượng điểm là việc cố gắng đưa ra một dự đoán 'tốt nhất' về tham số." |
| **`q10`** | Thủ thuật chọn hàm lõi (kernel trick) tương đương với việc thực hiện điều gì trên dữ liệu? | Tiền xử lý dữ liệu bằng hàm biến đổi phi tuyến | 51 | "Hàm lõi tương đương với việc tiền xử lý dữ liệu bằng hàm" |
| **`q11`** | Đầu ra của thuật toán k-NN đối với một điểm mới là gì? | Giá trị trung bình của các đầu ra ứng với các điểm lân cận gần nhất | 53 | "Đầu ra của thuật toán chính là giá trị trung bình của các đầu ra ứng với điểm gần nhất vừa tìm được." |
| **`q12`** | Điểm khác biệt cốt lõi của các thuật toán học không giám sát so với học có giám sát là gì? | Không có nhãn giám sát đi kèm dữ liệu | 56 | "học không giám sát không có sự hướng dẫn của nhãn mục tiêu" |
| **`q13`** | Khi số chiều của dữ liệu tăng lên, số lượng cấu hình cần xét đến có thể thay đổi như thế nào? | Tăng theo cấp số mũ | 66 | "Khi số chiều của dữ liệu tăng lên (từ trái qua phải), số lượng cấu hình cần xét đến có thể tăng theo cấp số mũ." |
| **`q14`** | Về mặt toán học, một đa tạp (manifold) được định nghĩa là gì trong không gian liên thông? | Tập hợp các điểm kết nối với nhau, có thể xấp xỉ cục bộ như một không gian Euclid | 71 | "Về mặt toán học, một đa tạp là một không gian topo cục bộ giống như không gian Euclid." |
| **`q15`** | Các thuật toán học có giám sát nói chung học mối liên kết giữa những yếu tố nào? | Học mối liên kết giữa đầu vào x và đầu ra y | 49 | "học mối liên kết giữa đầu vào x và mục tiêu y" |

### B. 2 câu thiếu bằng chứng (Unanswerable)

| ID | Câu hỏi | Lý do vắng mặt trong tài liệu (Absence Reason) | Hành vi kỳ vọng |
|---|---|---|:---:|
| **`q_unans_dev1`** | Thuật toán Transformer sử dụng cơ chế Multi-Head Attention như thế nào để xử lý chuỗi văn bản? | Chương 5 sách Deep Learning chỉ trình bày Machine Learning cơ bản; kiến trúc Transformer và cơ chế Multi-Head Attention hoàn toàn không xuất hiện trong tài liệu. | `search_and_abstain` |
| **`q_unans_dev2`** | Hàm kích hoạt Swish được định nghĩa theo công thức toán học nào và do nhóm tác giả nào đề xuất? | Hàm kích hoạt Swish do Google Brain đề xuất năm 2017, không nằm trong nội dung Chương 5; toàn bộ tài liệu không có từ khóa "Swish". | `search_and_abstain` |

---

## 2. Tập Test (7 câu)

### A. 5 câu có đáp án (Answerable)

| ID | Câu hỏi | Đáp án kỳ vọng | Trang | Bằng chứng (Gold Evidence) |
|---|---|---|:---:|---|
| **`q16`** | Các yếu tố quyết định để đánh giá một thuật toán ML có tốt hay không là khả năng của nó khi làm gì? | Làm cho sai số huấn luyện nhỏ và khoảng cách giữa sai số huấn luyện và kiểm thử là nhỏ | 18 | "Các yếu tố quyết định để đánh giá một thuật toán ML có tốt hay không là khả năng của nó khi làm cho sai số huấn luyện nhỏ và khoảng cách giữa sai số huấn luyện và sai số kiểm thử là nhỏ" |
| **`q17`** | Thủ thuật chọn hàm lõi trong SVM được đánh giá là một công cụ mạnh mẽ bởi mấy lý do? | Bởi hai lý do chính | 51 | "Thủ thuật chọn hãm lõi là một công cụ mạnh mẽ bởi hai lý do." |
| **`q18`** | Trong một tác vụ dịch máy, đầu vào được cung cấp cho hệ thống là gì? | Một chuỗi các ký tự trong một ngôn ngữ cụ thể | 5 | "Dịch máy: Trong một tác vụ dịch máy, đầu vào đã bao gồm một chuỗi các ký tự trong một số ngôn ngữ" |
| **`q19`** | Tại sao khi tính toán thống kê với một số lượng mẫu hữu hạn, kết quả ước lượng tham số lại có tính bất định? | Vì các mẫu khác nhau thu được từ cùng phân phối sẽ cho ra giá trị thống kê khác nhau | 36 | "Khi tính toán bất kỳ thống kê nào với một số lượng mẫu hữu hạn, kết quả ước lượng tham số của ta là bất định" |
| **`q20`** | Mục tiêu thúc đẩy sự phát triển các thuật toán Deep Learning là để vượt qua điều gì? | Vượt qua các trở ngại của thuật toán ML truyền thống trong các bài toán AI phức tạp | 3, 65 | "góp phần thúc đẩy sự phát triển các thuật toán DL nhằm vượt qua những trở ngại này." |

### B. 2 câu thiếu bằng chứng (Unanswerable)

| ID | Câu hỏi | Lý do vắng mặt trong tài liệu (Absence Reason) | Hành vi kỳ vọng |
|---|---|---|:---:|
| **`q_unans_test1`** | Cơ chế khuếch tán (diffusion) trong mô hình Denoising Diffusion Probabilistic Models (DDPM) hoạt động theo nguyên lý gì? | Mô hình sinh Diffusion / DDPM hoàn toàn không được đề cập trong Chương 5; từ khóa "diffusion" và "DDPM" có tần suất bằng 0. | `search_and_abstain` |
| **`q_unans_test2`** | Mạng nơ-ron tích chập ResNet giải quyết vấn đề triệt tiêu đạo hàm bằng residual connection như thế nào? | Kiến trúc mạng nơ-ron tích chập sâu với kết nối tắt (residual skip connection) không thuộc phạm vi Machine Learning cơ bản của Chương 5. | `search_and_abstain` |

---

## 3. Các File Lưu trữ Liên quan

- **File định dạng Markdown bảng tổng hợp:** [`data/BENCHMARK_DETAILS.md`](file:///c:/Users/Admin/Downloads/CODE/K5-L3B-Day07-2A202603972-NguyenCongHoan-RetrievalContext/data/BENCHMARK_DETAILS.md)
- **File JSON 24 ca benchmark:** [`data/agent_cases.json`](file:///c:/Users/Admin/Downloads/CODE/K5-L3B-Day07-2A202603972-NguyenCongHoan-RetrievalContext/data/agent_cases.json)
- **File JSON 20 câu có đáp án:** [`data/questions.json`](file:///c:/Users/Admin/Downloads/CODE/K5-L3B-Day07-2A202603972-NguyenCongHoan-RetrievalContext/data/questions.json)
- **Kết quả chạy thực tế của Agent trên Dev (17 câu):** [`results/agent/chosen_reranked_hybrid_dev/traces.json`](file:///c:/Users/Admin/Downloads/CODE/K5-L3B-Day07-2A202603972-NguyenCongHoan-RetrievalContext/results/agent/chosen_reranked_hybrid_dev/traces.json)
- **Kết quả chạy thực tế của Agent trên Test (7 câu):** [`results/agent/chosen_reranked_hybrid_test/traces.json`](file:///c:/Users/Admin/Downloads/CODE/K5-L3B-Day07-2A202603972-NguyenCongHoan-RetrievalContext/results/agent/chosen_reranked_hybrid_test/traces.json)
- **Điểm chấm chi tiết từng câu của người thẩm định:** [`results/agent/chosen_reranked_hybrid_test/reviews.json`](file:///c:/Users/Admin/Downloads/CODE/K5-L3B-Day07-2A202603972-NguyenCongHoan-RetrievalContext/results/agent/chosen_reranked_hybrid_test/reviews.json)

