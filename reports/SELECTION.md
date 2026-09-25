# Vì sao chọn lô này?

Trong 50 dòng đứng đầu `outputs/selection_round1.csv`, chọn năm frame bạn sẽ ưu tiên nếu chỉ có
ngân sách rà năm ảnh. Ghi tên, điểm, thời điểm, thứ tự và lý do; tối thiểu một quyết định phải xét
ảnh gần trùng hoặc trường hợp model không dự đoán được box: chọn `frame_0182.jpg` (điểm 0.9591,
72.8 s, hạng 1) vì có nhiều vùng mơ hồ (18) và là mẫu điểm cao nhất; `frame_0369.jpg` (0.9324,
147.6 s, hạng 2) vì có 43 box và 16 vùng mơ hồ; `frame_0380.jpg` (0.9170, 152.0 s, hạng 3)
để kiểm tra cảnh gần thời điểm với frame_0369; `frame_0326.jpg` (0.9155, 130.4 s, hạng 4)
vì có 39 box và 15 vùng mơ hồ; và `frame_0099.jpg` (0.9063, 39.6 s, hạng 8) để bổ sung
độ phủ thời gian. Cả năm frame đều được đánh dấu `selected=True` và đều có box; việc rà cặp
frame_0369/frame_0380 là kiểm tra gần trùng, tránh gán lặp cùng một cảnh.

Ba frame thuộc lô 12 ảnh model chọn và bằng chứng trong CSV/ảnh contact sheet: `frame_0182.jpg`,
`frame_0369.jpg` và `frame_0380.jpg`. CSV lần lượt xếp chúng hạng 1, 2, 3 với điểm 0.9591,
0.9324 và 0.9170; cột `selected` đều là `True`, `empty=False`, và số box lần lượt là 28,
43 và 40. Ba ảnh cũng xuất hiện trong contact sheet, cho phép đối chiếu trực tiếp các vùng
đối tượng và kiểm tra sự tương đồng của hai frame ở khoảng 147.6--152.0 giây.

Một frame có điểm cao nhưng không chọn hoặc một frame có điểm thấp vẫn nên xem, và lý do:
`frame_0372.jpg` có điểm 0.9101, hạng 6 nhưng `selected=False`; nên xem vì nằm giữa
`frame_0369.jpg` (147.6 s) và `frame_0380.jpg` (152.0 s), có 42 box và 15 vùng mơ hồ,
nhưng có thể là ảnh gần trùng nên không đưa vào ngân sách năm ảnh. Ngoài ra, `frame_0032.jpg`
(điểm 0.8203, hạng 50, 12.8 s, 31 box) là mẫu điểm thấp nhưng nên xem như một kiểm tra
đối chứng ở cuối danh sách, để phát hiện trường hợp điểm không phản ánh lỗi hoặc độ khó rõ ràng.

Điều phép chọn này chưa chứng minh về chất lượng mô hình: đây chỉ là chiến lược chọn mẫu dựa
trên điểm không chắc chắn, độ phủ thời gian và dấu hiệu gần trùng trong 50 dòng đầu. Nó không
cho biết precision, recall, mAP hay chất lượng box/class trên toàn bộ dữ liệu; cần nhãn chuẩn
độc lập và đánh giá trên tập kiểm tra chưa dùng để kết luận mô hình tốt hơn.
