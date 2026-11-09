# Phương pháp luận & Kiến trúc Hệ thống (Methodology)

Tài liệu này trình bày chi tiết cơ sở lý thuyết, quy trình trích xuất đặc trưng và kiến trúc mạng học sâu được sử dụng trong hệ thống Nhận diện Ngôn ngữ Ký hiệu Tiếng Việt (VSLR).

---

## 1. Biểu diễn Dữ liệu & Trích xuất Đặc trưng (Feature Representation)

Thay vì đưa trực tiếp ảnh RGB kích thước lớn vào mạng CNN 3D (gây tốn kém bộ nhớ và dễ bị nhiễu bởi phông nền, ánh sáng, trang phục), hệ thống sử dụng phương pháp tiếp cận dựa trên khung xương (**Pose-based Approach**).

### Cấu trúc Vector Đặc trưng (201 Features)
Trong mỗi khung hình (frame), hệ thống sử dụng tọa độ không gian 3 chiều $(x, y, z)$ của **67 điểm mốc (landmarks)** quan trọng:
* **Thân trên & Tay (Pose + Hands Landmarks):** Tổng cộng $67 	ext{ điểm} 	imes 3 	ext{ tọa độ } (x, y, z) = 201 	ext{ đặc trưng}$.
* Nếu một bộ phận bị khuất hoặc không xuất hiện trong khung hình, các tọa độ tương ứng được điền giá trị $0.0$ (Zero-padding) và được xử lý tự động bởi tầng `Masking` của mạng.

### Chuẩn hóa Chuỗi Thời gian (60-Frame Sequence)
Mỗi video cử chỉ có độ dài thực tế khác nhau được chuẩn hóa về độ dài cố định $T = 60$ khung hình thông qua phương pháp lấy mẫu tuyến tính đều (`np.linspace(0, total_frames - 1, 60)`). Nhờ đó, toàn bộ hành trình của cử chỉ từ lúc bắt đầu đến lúc kết thúc luôn được gói gọn trong ma trận đầu vào:
$$X \in \mathbb{R}^{60 	imes 201}$$

---

## 2. Kiến trúc Mạng Học sâu (Deep Learning Architecture)

Mô hình được thiết kế theo kiến trúc **Stacked Bidirectional LSTM với Temporal Attention**, cụ thể qua từng tầng như sau:

```text
Input Sequence: (60, 201)
       │
       ▼
┌────────────────────────────────────────────────┐
│ Masking Layer (mask_value = 0.0)               │
└──────────────────────┬─────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────┐
│ Bidirectional LSTM 1 (256 units, return_seq)   │ ──► Output: (60, 512)
│ + BatchNormalization + Dropout(0.3)            │
└──────────────────────┬─────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────┐
│ Bidirectional LSTM 2 (128 units, return_seq)   │ ──► Output: (60, 256)
│ + BatchNormalization + Dropout(0.3)            │
└──────────────────────┬─────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────┐
│ Custom Temporal Attention Layer                │ ──► Context Vector: (256,)
└──────────────────────┬─────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────┐
│ Dense (512, ReLU) + BatchNorm + Dropout(0.4)   │ ──► Output: (512,)
└──────────────────────┬─────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────┐
│ Output Dense Layer (Softmax, float32)          │ ──► Class Probabilities
└────────────────────────────────────────────────┘
```

### 2.1. Khối Stacked Bidirectional LSTM
Ngôn ngữ ký hiệu có tính phụ thuộc thời gian hai chiều rất rõ rệt. Việc xếp chồng 2 tầng BiLSTM (tầng đầu 256 units/chiều tạo ra 512 chiều ẩn; tầng hai 128 units/chiều tạo ra 256 chiều ẩn) giúp mô hình học được:
* **Ở tầng BiLSTM thứ nhất:** Các đặc trưng chuyển động cục bộ (vận tốc tay, hướng di chuyển ngắn hạn).
* **Ở tầng BiLSTM thứ hai:** Ngữ cảnh động học toàn cục của cả từ vựng qua chuỗi 60 frames.

### 2.2. Cơ chế Custom Temporal Attention
Đầu ra của tầng BiLSTM thứ hai là chuỗi trạng thái ẩn $H \in \mathbb{R}^{60 	imes 256}$. Tầng `Attention` tùy chỉnh thực hiện phép chiếu phi tuyến để tìm ra mức độ đóng góp của từng frame:

1. **Tính điểm năng lượng (Attention Score $e \in \mathbb{R}^{60 	imes 1}$):**
   $$e = 	anh(H W + b)$$
   Trong đó $W \in \mathbb{R}^{256 	imes 1}$ là ma trận trọng số và $b \in \mathbb{R}^{60 	imes 1}$ là vector độ lệch (bias) theo từng vị trí thời gian.

2. **Chuẩn hóa trọng số bằng Softmax theo trục thời gian (`axis=1`):**
   $$lpha_t = rac{\exp(e_t)}{\sum_{k=1}^{60} \exp(e_k)}$$

3. **Tổng hợp Vector Ngữ cảnh (Context Vector $c \in \mathbb{R}^{256}$):**
   $$c = \sum_{t=1}^{60} lpha_t h_t$$

---

## 3. Chiến lược Huấn luyện & Tối ưu hóa (Training & Optimization)

* **Data Pipeline (`tf.data.Dataset`):** Sử dụng cơ chế `from_generator` kết hợp `.shuffle(10000)`, `.batch(128)` và `.prefetch(tf.data.AUTOTUNE)` để nạp dữ liệu bất đồng bộ, tránh nghẽn cổ chai I/O.
* **Mixed Precision Training (`mixed_float16`):** Tăng tốc độ tính toán tensor và giảm một nửa dung lượng VRAM tiêu thụ, đồng thời ép kiểu tường minh `dtype="float32"` ở tầng `Softmax` cuối cùng để đảm bảo độ ổn định số học (numerical stability).
* **Hàm mất mát & Tối ưu:** Sử dụng `SparseCategoricalCrossentropy()` và trình tối ưu `Adam(learning_rate=1e-3)`.
* **Hệ thống Callbacks:**
  * `ModelCheckpoint`: Tự động lưu mô hình tốt nhất dựa trên `val_accuracy`.
  * `ReduceLROnPlateau`: Giảm một nửa learning rate (`factor=0.5`) nếu `val_loss` không giảm sau 2 epochs.
  * `EarlyStopping`: Dừng sớm nếu `val_accuracy` không cải thiện nhằm chống Overfitting.
