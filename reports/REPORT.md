# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Vũ Tiến Thăng

Công cụ gán nhãn đã dùng: CVAT

## 1. Dữ liệu và cách chia tập

Pool và tập kiểm thử được chia theo trục thời gian, có vùng đệm ở giữa, vì camera đứng yên và mỗi xe ở lại trong khung hình vài giây. Video được lấy mẫu 2,5 khung hình mỗi giây (`data/DATA.md`): hai ảnh cách nhau 0,4 giây gần như trùng nhau. Nếu chia ngẫu nhiên, cùng một chiếc xe có thể nằm ở cả tập train lẫn tập test. Mô hình khi đó được chấm trên xe nó đã thấy, nên AP50, precision và recall trên tập kiểm thử sẽ cao hơn năng lực thật trên cảnh mới. Đó là rò rỉ dữ liệu.

Cách chia trong bài tránh điều này: 20 ảnh test nằm ở bốn đoạn có tâm giây 20, 60, 100 và 140; 112 ảnh trong vùng 4 giây trước và sau mỗi đoạn bị loại; 268 ảnh còn lại là pool. Ảnh pool gần ảnh test nhất vẫn cách 4,4 giây.

## 2. Mô hình khởi đầu lạnh (cold start)

Dòng vòng 0 trong `reports/rounds_table.md`:

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

`outputs/metrics_round0.json` ghi cùng mức: AP50 0,7714; 197 TP, 16 FP, 206 FN trên 403 box tham chiếu (bỏ 14 box cao dưới 16 px). Lab chỉ chấm một lớp `car`, nên cold start không lệch theo từng lớp COCO car/bus/truck. Trên `outputs/compare_round0.jpg`, lệch nằm ở cỡ và khoảng cách: xe lớn gần camera thường có box xanh, còn xe nhỏ gần chân trời và xe chỉ còn đèn thì nhiều box vàng (bỏ sót) hoặc đỏ (không khớp tham chiếu). Bốn ảnh minh họa đều sót nhiều hơn là vẽ thừa: `frame_0050` TP 11 / FP 2 / FN 7, `frame_0150` TP 10 / FP 2 / FN 10, `frame_0250` TP 8 / FP 2 / FN 9, `frame_0350` TP 9 / FP 2 / FN 14.

Recall theo kích thước trong cùng file metrics: xe nhỏ 0,1818 (66 box), xe vừa 0,5473 (296 box), xe lớn 0,5610 (41 box). Cold start phủ được khoảng hơn một nửa xe vừa và xe lớn, nhưng gần như bỏ sót xe nhỏ. Precision 0,925 cho thấy khi model chịu vẽ box thì thường khớp tham chiếu; điểm yếu chính là bỏ sót.

Trước khi kết luận model sai, cần người rà nhãn tham chiếu trên `frame_0250`: cold start có box đỏ ở vùng xe nhìn thấy đèn, trong khi nhãn test do một model khác tạo và chưa được rà từng box (`data/DATA.md`). Xe chỉ còn hai chấm đèn gần chân trời cũng dễ lệch, vì guideline cho phép gán hoặc không khi box cao dưới khoảng 16 px và 14 box loại này đã bị bỏ qua khi chấm.

## 3. Chiến lược chọn mẫu

Điểm mỗi ảnh là `score = W_U·U + W_A·A + W_D·D` với trọng số 0,5 / 0,3 / 0,2 (`tools/al_select.py`). `U` là trung bình độ bất định của 5 box khó nhất; độ bất định của một box là `1 - |2·conf - 1|`, cao nhất khi confidence gần 0,5. `A` là số box mơ hồ (0,15 ≤ conf < 0,50), chia cho giá trị lớn nhất trong pool. `D` là khoảng cách thời gian tới ảnh đã gán gần nhất, chặn ở 10 giây rồi chia cho 10; vòng đầu chưa có ảnh nào đã gán nên `D = 1` cho mọi frame. `MIN_GAP_S = 2,0` giây: khi chọn tham lam theo điểm giảm dần, ảnh cách một ảnh vừa chọn dưới 2 giây bị bỏ, vì camera cố định nên hai khung quá gần gần như trùng và gán cả hai tốn công mà model học thêm rất ít.

Ba frame trong lô 12 ảnh, lấy từ `reports/SELECTION.md` và `outputs/selection_round1.csv`:

- `frame_0182.jpg`: điểm 0,9591, 72,8 s, hạng 1, `selected=True`. `U` 0,9182, `A` 1,0 (18 vùng mơ hồ trên 28 box conf ≥ 0,05), `D` 1,0. Đây là ảnh điểm cao nhất vì vừa phân vân vừa có nhiều box mơ hồ.
- `frame_0369.jpg`: điểm 0,9324, 147,6 s, hạng 2, 43 box và 16 vùng mơ hồ. Ảnh dày xe, đáng rà nếu ngân sách cho phép.
- `frame_0380.jpg`: điểm 0,9170, 152,0 s, hạng 3, 40 box và 15 vùng mơ hồ. Cách `frame_0369` 4,4 giây, vẫn đủ `MIN_GAP_S`, nên được giữ để kiểm tra cảnh gần thời điểm mà không trùng hẳn.

Frame cân nhắc thêm: `frame_0372.jpg`, điểm 0,9101, hạng 6, 148,8 s, 42 box, 15 vùng mơ hồ, nhưng `selected=False`. Nó nằm giữa `frame_0369` (147,6 s) và `frame_0380` (152,0 s), chỉ cách `frame_0369` 1,2 giây, dưới `MIN_GAP_S`. Điểm bất định gần bằng hai ảnh đã chọn, nhưng gán thêm sẽ lặp cùng một cảnh.

Điểm bất định không chứng minh ảnh đó sẽ cải thiện model. `U` cao chỉ nói confidence gần 0,5, không nói nhãn sau khi sửa sẽ làm AP50 tăng. Vòng 1 đã lấy đúng các frame điểm cao nhất, kể cả `frame_0182`, và AP50 vẫn giảm 0,242 so với cold start (`rounds_table.md`).

## 4. Các vòng học chủ động (active learning)

Bảng từ `reports/rounds_table.md`:

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 345 | 0.530 | -0.242 | 1.000 | 0.132 | 0.233 | 0.000 | 0.112 | 0.488 |

Vòng 1 train 12 ảnh, 345 box, 50 epoch (`outputs/metrics_round1.json`). `outputs/round1_diff.md`: model đề xuất 169 box, sau khi sửa còn 345 box; giữ nguyên 139, chỉnh 15, xóa 15, thêm 191; tỷ lệ chấp nhận pre-label 82%. Lỗi chính của nhãn gợi ý là bỏ sót (191 box thêm), không phải vẽ thừa (chỉ 15 box xóa).

AP50 vòng 1 là 0,530, giảm 0,242 so với cold start 0,771. Chỉ có một vòng fine-tune, nên mức giảm so với vòng trước cũng là 0,242. JSON ghi AP50 0,5297; tại conf 0,25 còn 53 TP, 0 FP, 350 FN. Precision lên 1,000 nhưng recall tụt từ 0,489 xuống 0,132.

Cùng tập test, mọi nhóm kích thước đều xấu đi. Xe nhỏ: recall 0,182 xuống 0,000 (66 box). Xe vừa: 0,547 xuống 0,112 (296 box), đây là nhóm kéo AP50 xuống mạnh nhất. Xe lớn: 0,561 xuống 0,488 (41 box), giảm ít nhất.

`outputs/compare_round1.jpg` cho một ca xấu đi có thể kiểm: `frame_0050` từ cold start TP 11 / FP 2 / FN 7 xuống vòng 1 TP 3 / FP 0 / FN 16. Model hết box đỏ nhưng chỉ còn vài box xanh trên xe lớn gần camera; xe giữa làn và xe xa chuyển thành bỏ sót. `frame_0350` cũng vậy: FN tăng từ 14 lên 20, FP từ 2 xuống 0. Khớp với precision 1,0 và recall 0,132: sau fine-tune, YOLOv8n trên 12 ảnh đêm trở nên quá thận trọng, không phải vẽ nhiều box giả.

Ba lớp bằng chứng không trộn vào nhau:

- Quan sát độc lập, trước pre-label: `reports/BLIND_SCAN.md` trên `frame_0099.jpg` đếm 26 xe. Hai chỗ dễ sót là xe sát mép bãi bên trái giữa, bị che một phần, và xe hàng cuối bên phải, thân ngắn lẫn nền. File được khóa trong `reports/blind_lock.json`.
- Lỗi pre-label đã sửa: cùng ảnh trong `outputs/round1_diff.md`, model đề xuất 13 box, sau sửa còn 26 box (khớp số đếm độc lập): giữ 11, sửa 1, xóa 1, thêm 14. Mười bốn box thêm là xe model bỏ sót; một box xóa là dự đoán không đủ thân xe; một box sửa là hộp không ôm sát.
- Kết quả sau train: `metrics_round1.json` và `compare_round1.jpg`. Số 345 box train không phải số đo trên test. AP50 0,530 đo mức khớp với nhãn tham chiếu sau fine-tune, không đo số box đã sửa.

Ca khó theo guideline, trên chính `frame_0099`: xe sát mép và bị che một phần. Guideline chỉ vẽ phần nhìn thấy, không suy nốt thân bị che, và xe cắt mép chỉ lấy phần nằm trong ảnh; hai xe sát nhau không gộp một box. Prelabel chỉ có 13 box trong khi quan sát độc lập thấy 26, nên các xe mép và xe bị che nằm trong 14 box thêm, không nằm trong kết quả model sau train.

## 5. Kết luận và giới hạn

Vòng 1 kém cold start trên tập test này: AP50 0,530 so với 0,771 (Δ −0,242), F1 0,233 so với 0,640. Model gần như không còn false positive (0 FP) nhưng bỏ sót 350/403 box, đặc biệt xe nhỏ và xe vừa. Tôi dừng, không train tiếp trên cùng chiến lược uncertainty và cùng cách fine-tune 50 epoch với 12 ảnh, cho đến khi kiểm tra nhãn và ngưỡng confidence. Train thêm ngay lúc recall đang sụp dễ làm model càng chỉ giữ xe lớn.

Nếu còn một vòng, hai ca lấy từ `outputs/selection_round2.csv`:

- Nên xem `frame_0067.jpg`: điểm 0,7750, 26,8 s, hạng 1, `U` 0,6999, `A` 0,75, `D` 1,0, 8 box và 6 vùng mơ hồ, `selected=True`. Cách `frame_0099` (39,6 s) khoảng 12,8 giây nên không trùng lô đã gán. Chi phí rà vẫn cao: vòng 1 cho thấy pre-label conf 0,25 chỉ có khoảng 13–20 box trong khi ảnh sau sửa có 22–37 box; 8 box gợi ý không phải số xe phải vẽ.
- Không gán `frame_0367.jpg` dù hạng 2, điểm 0,7015, `U` 0,9209. Thời điểm 146,8 s chỉ cách `frame_0369.jpg` đã gán (147,6 s) khoảng 0,8 giây, `D` chỉ 0,08. Camera cố định, hai ảnh gần như trùng; gán lại tốn công mà thêm rất ít thông tin.

Tập test chỉ 20 ảnh. `data/DATA.md` nói chênh dưới khoảng 0,01 AP50 chưa đủ để kết luận, nhưng mức giảm 0,242 lớn hơn ngưỡng đó nên có thể nói mức khớp với bộ tham chiếu này đã giảm. Kết luận vẫn không phải chân lý tuyệt đối: 14/417 box cao dưới 16 px bị bỏ qua, và toàn bộ nhãn tham chiếu do model tạo, chưa được người rà. Nếu tham chiếu sót xe hoặc vẽ lệch, AP50 đo sự đồng ý với bộ đó, không đo nhãn người.

AP50 đã giảm. Trước khi train thêm tôi sẽ kiểm tra, theo thứ tự: (1) 191 box thêm và 15 box sửa có bám guideline không, nhất là vệt đèn pha, xe che khuất và xe mép ảnh; (2) trên `compare_round1.jpg`, xe mất vì không còn box hay vì confidence dưới 0,25; (3) vài box tham chiếu trên `frame_0050` và `frame_0350` có phải xe thật không, trước khi coi mọi FN là lỗi model; (4) 50 epoch trên 12 ảnh với YOLOv8n có đang quên đặc trưng COCO, vì recall xe lớn giảm ít còn xe nhỏ về 0. Không sửa `data/test/labels/`.
