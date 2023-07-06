Các thư viện cần cài: Tensorflow, Numpy, Matplotlib, Python3

Data: Bộ dữ liệu data 438 nhân vật nặng Hơn 10GB nên không thể gửi kèm file được, thay vào đó là bộ dữ liệu đã qua xử lý hơn 300MB nằm trong VersionDatasetDubi

Thay vào đó sẽ dùng bộ data mẫu nhỏ đã có sẵn chỉ gồm 7 nhân vật.

Các bước thực hiện:

- Lấy dữ liệu từ nguồn dữ liệu thô chưa được xử lý (Có thể bỏ qua bước này đối với bộ 438 nhân vật)
B1: Run /source/Origin/getFace_fromDataOrigin.py
=> Kết quả thu được là các dataset khuôn mặt được cắt từ nguồn đầu vào => Lưu ở /dataset


-Train mô hình: Lúc này dữ liệu lấy từ dataset đã xử lý và huấn luyện cho mô hình (nếu muốn train bộ 438 thì di chuyển bộ dữ liệu dự bị vào /dataset và di chuyển bộ dữ liệu đang hiện có sang folder khác)

B2: Run source/Origin/train_FaceRecoginition.py
=> Kết quả thu được là mô hình được huấn luyện "model_faceRecoginition.v1": cho bộ dữ liệu nhỏ 7 nhân vật hoặc "model_faceRecoginition.v2": cho bộ dữ liệu lớn 438.

Quá trình train đã xong giờ qua phần nhận diện.

-Kiểm tra xem hiệu suất mô hình là bao nhiêu
B3: Run source/Origin/test_result.py
=> Kết quả thu được là số hình đúng với nhãn được gán (được hiển thị ở terminal)

B4: Run source/Origin/faceRecoginition.py 
=> Nhận diện ra nhân vật từ hình ảnh/video tuỳ vào path được đưa đến. Nếu đường dẫn sai => dữ liệu kiểm tra không tồn tại => Tự động kết nối với camera để lấy hình ảnh.

-Kết quả của quá trình train được lưu source/Origin/Picture.

-Ngoài ra: Muốn kiểm tra xem kết quả của hình ảnh vào sau từng layer sẽ chạy các file tương ứng trong /source/Layer và kết quả được lưu lại trong Picture cùng thư mục.


**Lưu Ý: Nếu muốn dùng bộ dữ liệu khác thì phải chú ý đến tên model đã train, và dataset tương ứng. Sai lệch sẽ đưa ra kết quả không chính xác.

Vì dung lượng file tối đa trong Teams là 500MB nên không thể gửi kèm dataOrigin. Lấy thêm dữ liệu tại: 
Github.com/hoaducviet/AI
