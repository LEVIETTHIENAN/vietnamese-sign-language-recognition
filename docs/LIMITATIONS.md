# Giới hạn hiện tại & Hướng phát triển (Limitations & Future Work)

Dưới đây là các phân tích kỹ thuật về giới hạn hiện tại của hệ thống VSLR và lộ trình nâng cấp trong tương lai.

---

## 1. Giới hạn hiện tại (Current Limitations)

### 1.1. Sự đồng bộ đặc trưng khi Suy luận Video thực tế (Inference Feature Alignment)
* Bộ dữ liệu huấn luyện gốc gồm **201 đặc trưng** (kết hợp cả khớp thân người và các khớp ngón tay). 
* Trong pipeline suy luận thử nghiệm với `pose_landmarker_lite.task` (chỉ trả về 33 điểm mốc cơ thể $	imes 3 = 99$ số), việc đệm thêm các số $0$ (zero-padding từ 99 lên 201) giúp mô hình chạy được về mặt kích thước tensor nhưng làm mất đi 102 đặc trưng chi tiết của hai bàn tay. Để đạt hiệu năng tối đa khi chạy thực tế, pipeline suy luận cần kết hợp đồng thời cả `PoseLandmarker` và `HandLandmarker` (hoặc `MediaPipe Holistic`).

### 1.2. Hiện tượng tự che khuất (Self-Occlusion) và Góc máy
* Khi hai bàn tay đan vào nhau hoặc xoay vuông góc với ống kính camera, MediaPipe có thể bị mất dấu các khớp ngón tay hoặc dự đoán sai lệch tọa độ chiều sâu $z$.
* Tọa độ $(x, y, z)$ hiện tại phụ thuộc vào vị trí đứng của người thực hiện trong khung hình. Nếu người dùng đứng quá lệch sang một góc hoặc quá xa camera, phân phối tọa độ đầu vào sẽ bị dịch chuyển.

### 1.3. Thiếu hụt các yếu tố phi thủ công (Non-Manual Markers)
* Trong Ngôn ngữ Ký hiệu Tiếng Việt (VSL), ý nghĩa của từ còn phụ thuộc vào **biểu cảm khuôn mặt** (nhíu mày, hình dáng miệng) và **chuyển động đầu** (để phân biệt câu hỏi, phủ định, cảm xúc).
* Vector 201 chiều hiện tại chưa tích hợp sâu các điểm mốc chi tiết trên cơ mặt (Face Mesh).

### 1.4. Giới hạn ở Nhận diện Từ đơn (Isolated Sign Recognition)
* Hệ thống đang giải quyết bài toán phân loại từng ký hiệu rời rạc trong cửa sổ 60 frames cố định. Khi áp dụng vào giao tiếp liên tục (Continuous Sign Language Recognition), việc tự động phân đoạn ranh giới giữa các từ liên tiếp cần thêm cơ chế cửa sổ trượt (Sliding Window) hoặc giải mã CTC (Connectionist Temporal Classification).

---

## 2. Hướng phát triển tương lai (Future Work)

1. **Hoàn thiện Holistic Extractor cho Inference:**
   Kết hợp đầy đủ `PoseLandmarker` + `HandLandmarker` trong hàm `extract_201()` để lấp đầy toàn bộ 201 đặc trưng thực tế khi chạy qua webcam hoặc video mới.
2. **Chuẩn hóa tọa độ bất biến (Viewpoint & Scale Normalization):**
   Đưa toàn bộ tọa độ $(x, y, z)$ về hệ quy chiếu lấy tâm là điểm giữa hai vai và chia tỷ lệ theo khoảng cách hai vai, giúp mô hình hoạt động ổn định bất chấp khoảng cách từ người dùng tới camera.
3. **Mở rộng sang Dịch thuật Ngôn ngữ Ký hiệu (Sign Language Translation - SLT):**
   Kết hợp đầu ra từ vựng của mạng BiLSTM-Attention với một mô hình ngôn ngữ lớn (LLM) tiếng Việt để tự động sắp xếp lại chuỗi từ vựng VSL thành câu tiếng Việt hoàn chỉnh, đúng ngữ pháp.
