# ĐỒ ÁN CUỐI KỲ – NHÓM 23  
## Phân tích hành vi sử dụng điện thoại thông minh và mức độ nghiện ở thanh thiếu niên

---

## 1. Tổng quan dự án & Thông tin nhóm

### 1.1. Mục tiêu dự án
Dự án tập trung phân tích **hành vi sử dụng điện thoại thông minh** và **mức độ nghiện điện thoại** ở thanh thiếu niên (13–19 tuổi), đồng thời khám phá mối liên hệ giữa việc sử dụng điện thoại với:
- Giấc ngủ
- Kết quả học tập
- Sức khỏe tâm thần
- Mối quan hệ gia đình và xã hội  

Bên cạnh phân tích thống kê và trực quan hóa dữ liệu, nhóm xây dựng **mô hình học máy** nhằm dự đoán mức độ nghiện điện thoại dựa trên các hành vi sử dụng thường ngày.

### 1.2. Thông tin thành viên
- **23120151** – Huỳnh Yến Nhi  
- **23120152** – Hồ Khổng Tuyết Như  
- **23120172** – Trần Thị Thủy Tiên  

---

## 2. Dataset – Nguồn dữ liệu và mô tả

### 2.1. Nguồn dữ liệu
- **Nền tảng:** Kaggle  
- **Tên dataset:** Teen Smartphone Usage and Addiction Impact Dataset  
- **Link:** https://www.kaggle.com/datasets/sumedh1507/teen-phone-addiction  
- **Tác giả:** sumedh1507  
- **Thời gian cập nhật:** Khoảng 5 tháng trước

### 2.2. Cấp phép
Dataset được cấp phép **CC0 – Creative Commons Zero**, cho phép sử dụng, chỉnh sửa và phân tích tự do cho mục đích học tập và nghiên cứu.

### 2.3. Phương pháp thu thập
- Hình thức: Khảo sát  
- Đối tượng: Học sinh trung học và sinh viên năm nhất (13–19 tuổi)  
- Thời gian thu thập: Khoảng 3 tháng  
- Khu vực: Trường học ở khu vực thành thị và bán thành thị  

### 2.4. Quy mô và chất lượng dữ liệu
- **3000 dòng × 25 cột**
- Không có dòng trùng lặp
- Không có dòng trống hoàn toàn
- Không có missing values
- Có outliers nhưng ở mức hợp lí
- Dataset sạch, phù hợp cho phân tích và xây dựng mô hình

---

## 3. Câu hỏi nghiên cứu (Research Questions)

1. Có phải thanh thiếu niên càng ít giao tiếp với gia đình thì càng dễ nghiện điện thoại hơn không?  
2. Mức độ lo âu/trầm cảm có ảnh hưởng đến hành vi sử dụng điện thoại của thanh thiếu niên không?  
3. Mục đích sử dụng điện thoại nào (mạng xã hội, chơi game, học tập) liên quan mạnh nhất đến mức độ nghiện?  
4. Thời gian sử dụng điện thoại trước khi ngủ ảnh hưởng như thế nào đến chất lượng giấc ngủ, kết quả học tập và sức khỏe tinh thần của thanh thiếu niên?  
5. Nhóm nghiện điện thoại thấp khác biệt ra sao so với nhóm nghiện cao về các yếu tố hành vi, tâm lý và xã hội?  
6. Có thể dự đoán mức độ nghiện điện thoại của một thanh thiếu niên dựa trên các hành vi sử dụng hằng ngày hay không?

---

## 4. Các phát hiện chính (Key Findings)

### 4.1. Phát hiện từ phân tích dữ liệu
- `Daily_Usage_Hours` là yếu tố ảnh hưởng mạnh nhất đến `Addiction_Level`.
- Thời gian sử dụng **mạng xã hội** (`Time_on_Social_Media`) và **chơi game**(`Time_on_Gaming`) là hai biểu hiện rõ rệt nhất của hành vi nghiện.
- Sử dụng điện thoại trước khi ngủ có thể làm giảm thời gian ngủ và liên quan đến mức độ lo âu cao hơn.
- Giao tiếp gia đình tốt và tập thể dục thường xuyên có thể đóng vai trò là **yếu tố bảo vệ**.

### 4.2. Kết quả mô hình học máy
- So sánh **Linear Regression** và **Random Forest Regression**:
  - Random Forest vượt trội hơn trên mọi metric (MAE, RMSE, R²).
  - R² đạt **0.7138**, phản ánh khả năng dự đoán tốt.
- Mô hình sử dụng 6 features chính (chủ yếu dự đoán dựa trên hành vi sử dụng điện thoại hàng ngày):
  - `Daily_Usage_Hours`
  - `Time_on_Gaming`
  - `Time_on_Social_Media`
  - `Phone_Checks_Per_Day`
  - `Time_on_Education`
  - `Screen_Time_Before_Bed`

---

## 5. Cấu trúc thư mục dự án

```text
final-project-CSC17104/
│
├── data/
│   └── raw/                              # Dữ liệu gốc
│       └── teen_phone_addiction.csv
│
├── notebook/                             # Jupyter Notebook phân tích
│   └── Group_23.ipynb
│
├── src/                                  # Source code Python
│   ├── data_processing.py               # Tiền xử lý dữ liệu
│   ├── models.py                        # Xây dựng & huấn luyện mô hình
│   └── visualization.py                # Trực quan hóa dữ liệu
│
├── document/                             
│   └── document.pdf
|
├── README.md                             # Mô tả dự án
└── requirements.txt                     # Danh sách thư viện
```
---
## 6. Hướng dẫn chạy dự án (How to Run)

### 6.1. Yêu cầu môi trường
- **Python**: >= 3.9
- **Jupyter Notebook**
- **Khuyến nghị**: Sử dụng môi trường ảo (`venv` hoặc `conda`)

---

### 6.2. Cài đặt

#### **Bước 1: Clone repository**
```bash
git clone https://github.com/insenseemarch/final-project-CSC17104.git
cd final-project-CSC17104
```

#### **Bước 2: Tạo và kích hoạt môi trường ảo (khuyến nghị)**

**Với `venv` (Python built-in):**
```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt (Windows)
venv\Scripts\activate

# Kích hoạt (Linux/macOS)
source venv/bin/activate
```

**Với `conda`:**
```bash
# Tạo môi trường
conda create -n min_ds-env python=3.9

# Kích hoạt
conda activate min_ds-env
```

#### **Bước 3: Cài đặt thư viện**
```bash
pip install -r requirements.txt
```

> **Lưu ý**: File `requirements.txt` bao gồm:
> - numpy
> - pandas
> - matplotlib
> - seaborn
> - scikit-learn
> - jupyter

---

### 6.3. Chạy dự án

#### **Bước 1: Khởi động Jupyter Notebook**
```bash
jupyter notebook
```

#### **Bước 2: Mở file notebook**
Trong giao diện Jupyter, điều hướng đến:
```
notebook/Group_23.ipynb
```

#### **Bước 3: Chạy notebook**
Thực thi các cell theo thứ tự từ **trên xuống dưới** (hoặc chọn `Cell > Run All`).

---

## 7. Dependencies

Dự án sử dụng các thư viện Python sau:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- jupyter

Danh sách đầy đủ được khai báo trong file `requirements.txt`.

---

