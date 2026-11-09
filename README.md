# Vietnamese Sign Language Recognition (VSLR) 🤟🇻🇳

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose%20%26%20Holistic-green.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

Hệ thống nhận diện Ngôn ngữ Ký hiệu Tiếng Việt (Vietnamese Sign Language - VSL) tự động dựa trên chuỗi hình ảnh động theo thời gian thực. Dự án sử dụng pipeline trích xuất khung xương không gian kết hợp với mạng nơ-ron học sâu **Bidirectional LSTM (BiLSTM)** tích hợp **Attention Mechanism** nhằm hỗ trợ giao tiếp cho cộng đồng người khiếm thính tại Việt Nam.

---

## 🌟 Điểm nổi bật (Key Features)

* **Trích xuất đặc trưng tối ưu (Pose-based Feature Extraction):** Chuyển đổi video thô thành chuỗi tọa độ khung xương không gian $(x, y, z)$, giúp giảm thiểu chi phí tính toán và loại bỏ nhiễu từ môi trường nền.
* **Kiến trúc BiLSTM 2 tầng kết hợp Mixed Precision (`mixed_float16`):** Nắm bắt ngữ cảnh chuyển động cả hai chiều thuận/nghịch với tốc độ huấn luyện tối ưu trên GPU.
* **Cơ chế Temporal Attention tùy chỉnh (Custom Attention Layer):** Tự động đánh trọng số cao cho các khung hình mang tính quyết định ("key frames") của cử chỉ và bỏ qua các khung hình nghỉ hoặc chuyển giao.
* **Xử lý chuỗi cố định:** Chuẩn hóa dữ liệu đầu vào theo chuỗi **60 khung hình (frames)** liên tiếp với **201 đặc trưng không gian (pose features)** trên mỗi khung hình.

---

## 📂 Cấu trúc dự án (Repository Structure)

```text
Vietnamese-Sign-Language-Recognition/
├── README.md                          # Tổng quan, kiến trúc và hướng dẫn dự án
├── LICENSE                            # Giấy phép mã nguồn mở (MIT)
├── requirements.txt                   # Danh sách các thư viện sử dụng
├── .gitignore                         # Cấu hình bỏ qua các file rác/tạm thời
├── notebooks/
│   ├── vsl_bilstm_attention.ipynb     # Notebook huấn luyện, đánh giá và suy luận
│   └── training_log.csv               # Nhật ký quá trình huấn luyện qua từng epoch
├── docs/
│   ├── METHODOLOGY.md                 # Giải thích chi tiết kiến trúc và toán học
│   └── LIMITATIONS.md                 # Phân tích giới hạn và hướng phát triển
└── models/
    └── best_model.keras               # Trọng số mô hình đã huấn luyện hoàn chỉnh
```

---

## 🚀 Hướng dẫn sử dụng nhanh (Quick Start)

Dự án đã được huấn luyện và đóng gói sẵn trọng số tối ưu tại `models/best_model.keras`. Bạn có thể chạy suy luận (inference) trực tiếp trên Google Colab, Kaggle hoặc máy cá nhân mà không cần huấn luyện lại từ đầu.

### 1. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 2. Tải mô hình và Dự đoán (Inference)
Do mô hình sử dụng tầng `Attention` tùy chỉnh, khi tải `best_model.keras` cần khai báo `custom_objects`:

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

# 1. Định nghĩa tầng Custom Temporal Attention
class Attention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        score = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * x, axis=1)
        return context

# 2. Tải mô hình đã huấn luyện
model = tf.keras.models.load_model(
    "models/best_model.keras",
    custom_objects={"Attention": Attention}
)

# 3. Dự đoán trên chuỗi đặc trưng (batch_size=1, frames=60, features=201)
sample_sequence = np.random.rand(1, 60, 201).astype(np.float32)
predictions = model.predict(sample_sequence, verbose=0)[0]
top_class_idx = np.argmax(predictions)
print(f"Predicted Class ID: {top_class_idx} (Confidence: {predictions[top_class_idx]:.4f})")
```

---

## 🧠 Tổng quan phương pháp (Methodology Overview)

Hệ thống hoạt động qua 3 giai đoạn chính:
1. **Computer Vision Pose-Detection Pipeline:** Thu nhận video và lấy mẫu đều **60 frames** (`np.linspace`), trích xuất tọa độ 3D $(x, y, z)$ của các khớp xương tạo thành vector **201 chiều** cho mỗi frame.
2. **Sequence Masking & BiLSTM:** Chuỗi $(60 	imes 201)$ đi qua tầng `Masking(0.0)` để bỏ qua các vùng đệm, sau đó qua 2 khối `Bidirectional LSTM` (256 units và 128 units) kèm `BatchNormalization` và `Dropout(0.3)`.
3. **Attention & Classification Head:** Tầng `Attention` tổng hợp toàn bộ chuỗi thời gian thành một vector ngữ cảnh duy nhất, truyền qua tầng `Dense(512, activation="relu")` và phân loại đầu ra bằng `Softmax`.

👉 *Xem chi tiết toán học và kiến trúc tại [docs/METHODOLOGY.md](docs/METHODOLOGY.md).*

---

## ⚠️ Giới hạn & Hướng phát triển
Dự án hiện tập trung vào nhận diện ký hiệu từ vựng độc lập dựa trên chuyển động tay và tư thế thân trên. Một số yếu tố như sự bất đồng bộ giữa bộ trích xuất khi suy luận, biểu cảm khuôn mặt (non-manual markers) hay dịch thuật câu liên tục được phân tích kỹ để mở rộng trong tương lai.

👉 *Xem phân tích chi tiết tại [docs/LIMITATIONS.md](docs/LIMITATIONS.md).*

---

## 📜 Giấy phép (License)
Dự án này được phân phối dưới giấy phép MIT. Xem file [LICENSE](LICENSE) để biết thêm chi tiết.
