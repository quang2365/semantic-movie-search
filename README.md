<<<<<<< HEAD
 
=======
# 🎬 Semantic Movie Search Engine (Hybrid Retrieval)

Hệ thống tìm kiếm phim thông minh sử dụng kỹ thuật truy hồi lai (Hybrid Retrieval) kết hợp giữa từ khóa (BM25) và ngữ nghĩa (Vector Search). Hệ thống cho phép tìm kiếm phim dựa trên mô tả nội dung bằng tiếng Anh với độ chính xác cao.

---

## 🏗️ Kiến trúc & Quy trình xử lý

Hệ thống chia làm 2 giai đoạn chính:

### 1. Luồng Offline (Xây dựng Knowledge Base)

- **Ingest**: Thu thập dữ liệu từ TMDB API (bao gồm thông tin phim và Poster URL).
- **Process**: Làm sạch dữ liệu và chia nhỏ (Chunking) nội dung.
- **Embedding**: Chuyển đổi văn bản thành Vector bằng mô hình `all-MiniLM-L6-v2`.
- **Indexing**: Lưu trữ Vector vào FAISS, văn bản vào BM25 và thông tin chi tiết vào SQLite.

### 2. Luồng Online (Truy hồi thời gian thực)

- **Query Processing**: Chuẩn hóa câu hỏi và chuyển thành Vector.
- **Hybrid Retrieval**: Tìm kiếm song song trên BM25 (từ khóa) và FAISS (ngữ nghĩa).
- **RRF & MaxP**: Trộn kết quả và gom nhóm các đoạn văn về bộ phim gốc.
- **Cross-Encoder Reranking**: Chấm điểm lại Top 30 phim để lấy kết quả liên quan nhất.

---

## 🚀 Hướng dẫn vận hành hệ thống

### Bước 1: Chuẩn bị môi trường

Yêu cầu Python 3.9 trở lên. Clone dự án và cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

### Bước 2: Cấu hình API Key

Tạo file .env tại thư mục gốc và dán mã API TMDB của bạn vào:

Đoạn mã

# TMDB_API_KEY= "fad932ee6979979b582abbc95e1c96bf"

### Bước 3: Quy trình chạy Offline (Khởi tạo dữ liệu)

Đây là bước quan trọng để xây dựng bộ não cho hệ thống. Bạn chỉ cần chạy file tổng hợp:

Bash
python run_offline.py

Lưu ý: Quá trình này sẽ mất vài phút để tải dữ liệu và nhúng Vector. Sau khi chạy xong, các tệp chỉ mục sẽ xuất hiện trong thư mục index/ và data/.

### Bước 4: Chạy ứng dụng Web (Giao diện người dùng)

Sau khi đã có dữ liệu ở Bước 3, khởi động giao diện Streamlit:

Bash
python -m streamlit run app.py

semantic-movie-search/
├── data/ # Lưu trữ file CSV thô và sạch sau khi xử lý
├── index/ # Chứa các chỉ mục tìm kiếm (FAISS, BM25, SQLite)
├── pipeline/ # Mã nguồn các module xử lý Offline (M1 -> M5)
├── retrieval/ # Mã nguồn các module truy hồi Online (M6 -> M13)
├── ui/ # Thành phần giao diện Streamlit
├── .env # File cấu hình biến môi trường (API Key)
├── app.py # File chạy chính của ứng dụng
├── requirements.txt # Danh sách thư viện cần cài đặt
└── README.md # Hướng dẫn sử dụng

