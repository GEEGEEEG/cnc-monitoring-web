# CNC Milling Machine Failure Prediction Web App ⚙️

Ứng dụng web dự đoán tình trạng hư hỏng máy phay CNC công nghiệp dựa trên dữ liệu vận hành (Giai đoạn GĐ7 - GĐ10).

## 🚀 Hướng dẫn đưa lên GitHub và mở Web trong 2 phút

### Bước 1: Tạo Repository trên GitHub
1. Đăng nhập [GitHub](https://github.com) -> Chọn **New repository**.
2. Đặt tên (ví dụ: `cnc-failure-prediction`) -> Để chế độ **Public** -> Bấm **Create repository**.

### Bước 2: Kéo thả các file vào GitHub
Tải lên repo các file trong thư mục này:
- `app.py` (Mã nguồn web chính)
- `requirements.txt` (Khai báo thư viện)
- `best_model.joblib` (File mô hình tải từ Google Drive của bạn: `/content/drive/MyDrive/CNC_ML_Project/models/best_model.joblib`)

### Bước 3: Xuất Web trên Streamlit Cloud (Miễn phí)
1. Truy cập [share.streamlit.io](https://share.streamlit.io) và đăng nhập bằng tài khoản GitHub.
2. Bấm **"New app"**.
3. Chọn Repository `cnc-failure-prediction`, nhánh `main`, file `app.py`.
4. Bấm **Deploy!**

Sau 1-2 phút, bạn sẽ có một link web chính thức (dạng `https://cnc-failure-prediction.streamlit.app`) để đưa vào báo cáo và demo!
