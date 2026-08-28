# SARSA chơi Snake — tài liệu thuyết trình

Đồ án: huấn luyện agent **tabular SARSA** chơi Snake trên lưới 10×10 và 30×30.  
Code: `common/snake_env.py`, `agents/sarsa/`, `app.py`.

Policy lúc train / eval / chơi đều lấy từ Q-table. Reward shaping nằm trong môi trường; **không** ghi đè action bằng heuristic lúc chơi.

---

## 1. Bài toán MDP (tóm tắt 1 slide)

| Thành phần | Trong đồ án |
|---|---|
| Tác nhân (agent) | Tabular SARSA, bảng Q |
| Môi trường (env) | Snake Gymnasium, lưới `n×n`, `n ∈ {10, 30}` |
| Trạng thái (state) | Vector 7 chiều **tương đối**, **11.520** ô |
| Hành động (action) | 4 hướng tuyệt đối: lên / phải / xuống / trái |
| Phần thưởng (reward) | Ăn mồi, chết, A* gần/xa mồi, giữ đuôi, cắt đường mồi |
| Mục tiêu | Điểm càng cao càng tốt; thắng khi fill hết bàn |

Thắng:

- 10×10: score **97** (100 − 3)
- 30×30: score **897** (900 − 3)

Train mặc định trên **10×10**. Cùng Q-table chơi được 30×30 vì state không chứa kích thước lưới.

---

## 2. Môi trường

- Chuẩn **Gymnasium**: `reset()`, `step(action) → obs, reward, terminated, truncated, info`
- Rắn bắt đầu dài 3, giữa bàn, hướng phải
- Thức ăn spawn ngẫu nhiên trên ô trống
- Bước ngược hướng hiện tại bị **đổi thành đi thẳng** (luật game, không phải policy agent)
- Cắt episode: `max_steps` (10×10: 2000; 30×30: 18.000 theo diện tích), hoặc `no_food` (10×10: 100 bước; 30×30: 300)

Kết thúc ván (`end_reason`): `win` | `death` | `no_food` | `max_steps`

---

## 3. Hành động

Không gian hành động: **4** (Discrete)

| Mã | Hướng |
|---|---|
| 0 | UP |
| 1 | RIGHT |
| 2 | DOWN |
| 3 | LEFT |

| Giai đoạn | Policy |
|---|---|
| **Train** | ε-greedy trên Q-table |
| **Eval** | greedy: `argmax_a Q(s, a)` |
| **Chơi** (`app.py`) | cùng greedy Q |

Agent học policy; eval đo đúng policy đó. Không có bộ lọc / bám đuôi / lookahead ghi đè nước đi.

---

## 4. Trạng thái (liệt kê từng chiều)

Observation: `MultiDiscrete([3, 3, 4, 4, 4, 4, 5])`

**Tổng số trạng thái tabular:**

\[
3 \times 3 \times 4 \times 4 \times 4 \times 4 \times 5 = 11\,520
\]

Đây là **feature**, không phải toàn bộ bàn cờ (bàn 10×10 đầy đủ sẽ khổng lồ; 30×30 còn lớn hơn).

| # | Tên | Số giá trị | Ý nghĩa |
|---|---|---|---|
| 1 | `food_dx` | 3 | Thức ăn trái / cùng cột / phải (`sign(x_food − x_head) + 1`) |
| 2 | `food_dy` | 3 | Thức ăn trên / cùng hàng / dưới |
| 3 | `safety_straight` | 4 | An toàn nếu **đi thẳng** |
| 4 | `safety_left` | 4 | An toàn nếu **rẽ trái** |
| 5 | `safety_right` | 4 | An toàn nếu **rẽ phải** |
| 6 | `direction` | 4 | Hướng đầu rắn hiện tại (0–3) |
| 7 | `fill_bucket` | 5 | Mật độ bàn: `min(4, int(len(snake)/n² × 5))` — mỗi bậc ~20% ô |

### 4.1. Bốn mức safety (BFS / flood-fill)

Mỗi hướng thẳng / trái / phải: giả lập **một bước**, rồi BFS.

| Giá trị | Tên | Ý nghĩa |
|---|---|---|
| 0 | COLLISION | Đâm tường hoặc thân ngay |
| 1 | TRAP | Số ô tới được &lt; độ dài rắn → không đủ chỗ chứa thân |
| 2 | TIGHT | Đủ chỗ nhưng **mất đường tới đuôi** |
| 3 | OPEN | Còn BFS tới đuôi → có thể bám đuôi mà sống |

Khi **ăn**, đuôi **không rút** — safety tính đúng việc đó.

### 4.2. Vì sao 11.520 chứ không phải “mọi thế cờ”?

State **tương đối**: cùng “mồi bên phải + rẽ trái thì thoáng + đầy 20% bàn” dùng chung cho 10×10 và 30×30.  
Đổi: trước đây `length_bucket = score // 5` (score tuyệt đối không chuyển được giữa hai kích thước lưới). Đã đổi sang **mật độ**.

---

## 5. Tác nhân — tabular SARSA

Cập nhật on-policy:

\[
Q(s,a) \leftarrow Q(s,a) + \alpha \big[ r + \gamma Q(s',a') - Q(s,a) \big]
\]

`a'` là hành động **thực sự** bước tiếp (ε-greedy), không phải `max Q` như Q-learning.

| Tham số | Giá trị mặc định | Vai trò |
|---|---|---|
| α (alpha) | 0.1 | Tốc độ học |
| γ (gamma) | 0.99 | Coi trọng thưởng tương lai |
| ε start | 1.0 | Ban đầu 100% explore |
| ε min | 0.01 | Sàn explore |
| ε decay | 0.9998 | Nhân mỗi episode (~chạm sàn khoảng ep 15k) |
| Episodes | 30.000 | |
| Seed train | 42 | Lặp lại được lần train |
| Eval | mỗi 1000 ep × 100 ván greedy | Lưu `agent_best.pkl` nếu mean score tăng |

Train: `python -m agents.sarsa.train`  
Chơi: `python app.py` — menu tick **Load model** rồi chọn 10 / 30; bỏ tick = random. `--seed N` để replay. `--grid 10` vào thẳng màn (load model); thêm `--random` nếu không load.

**Lưu ý slide:** ε decay quá chậm (ví dụ 0.99997) → hết 30k episode ε vẫn ~0.4 → SARSA học policy gần như random. Phải để ε kịp giảm trong 30k episode.

Q-table khởi tạo **0**. Eval greedy lúc đầu ≈ random (thường ~0–1). Nếu eval đã ~45 từ episode 1000 thì đó là heuristic ghi đè, không phải SARSA đã học.

---

## 6. Phần thưởng (reward shaping)

Giá trị mặc định trong `env_hyperparameters.py`. Shaping **chỉ** đổi số `r` trong công thức SARSA; agent vẫn tự chọn action.

| Tín hiệu | Giá trị | Khi nào |
|---|---|---|
| Ăn thức ăn | **+10** | Đầu trùng ô mồi |
| Chết | **−20 − 0.5 × score** | Đâm tường/thân (chết lúc điểm cao phạt nặng hơn) |
| Mỗi bước | **−0.001** | Khuyến khích đi ngắn |
| Gần mồi (A*) | **+0.02 × scale × Δ** | A* tới mồi giảm |
| Đi vòng (không đúng 1 ô A*) | **−0.05 × scale** | Detour |
| A* mất đường | **−0.2** | Mồi bị thân chắn sẵn |
| Vừa cắt đường tới mồi | **−0.8** | Trước bước còn A*, sau bước mất |
| Còn đường tới đuôi | **+0.05** | Giữ hành lang sống |

`scale` (A*): **1.0** khi `fill_bucket < 1` (rắn &lt; ~20% bàn); sau đó còn **0.15** — bớt ép đuổi đường ngắn nhất khi rắn dài.

A* = đường ngắn nhất 4 hướng, **né thân** (không dùng Manhattan xuyên thân).

---

## 7. Policy lúc train / eval / chơi

Cùng một hàm `SarsaAgent.choose_action`:

1. **Train:** với xác suất ε chọn đều 4 hướng; còn lại `argmax Q`.
2. **Eval / chơi:** luôn `argmax Q` (`greedy=True`).
3. Môi trường biến 180° thành đi thẳng — luật vật lý, áp dụng mọi policy.

**Không làm lúc chơi:** bám đuôi bắt buộc, cấm đớp túi, lookahead, lọc nhóm OPEN. Những việc đó từng làm `eval_score` trông cao nhưng **không** phải policy agent học được.

UI: bàn 10×10 ô lớn; bàn 30×30 thu ô cho vừa màn hình, nền caro, FPS cao hơn vì ván dài.

---

## 8. Vấn đề mắc kẹt và cách giải

Dùng bảng này cho slide “Khó khăn & giải pháp”.

| # | Vấn đề quan sát được | Nguyên nhân | Cách giải (thuật toán / kỹ thuật) |
|---|---|---|---|
| 1 | Train score không lên, eval kẹt thấp | `ε_decay` quá chậm → SARSA học policy gần random | `ε_decay = 0.9998` để ε chạm sàn trong ~15k episode |
| 2 | Eval lúc đầu đã ~45, gần như không tăng | Heuristic ghi đè greedy Q lúc eval/chơi | Gỡ heuristic; eval = greedy Q. Điểm đầu về ~0–1 |
| 3 | Đi vòng hết bước / `no_food` | State lặp, greedy đi hoài một vòng | Phạt bước + timeout `no_food`; agent phải học thoát |
| 4 | Cùng score tuyệt đối khác nghĩa trên hai lưới | `score//5` không chuyển được 10×10 ↔ 30×30 | **`fill_bucket` = mật độ `len/n²`** |
| 5 | Ăn xong kẹt ngay | Safety chỉ nhìn **1 bước**; đuôi không rút khi ăn | Feature safety + phạt chết theo score; không lookahead lúc chơi |
| 6 | Mồi bị thân bao | State 7 chiều không thấy túi kín trên bàn | Phạt A* mất đường / cắt đường; agent học né từ reward |
| 7 | Hai ván giống hệt, tưởng random | `app.py --seed 42` + greedy → cùng ván | Mỗi ván một seed; `--seed N` chỉ khi cần replay |
| 8 | Fill một phần bàn rồi chết, 0 win | Tabular không encode hình bàn → không packing tới 97/897 | Giới hạn của feature; **không** vá bằng Hamiltonian / bám đuôi lúc chơi |

**Không làm:** Hamiltonian cycle hay heuristic chơi hộ — policy lúc đó không còn là SARSA.

---

## 9. Số liệu minh họa (để ghi slide kết quả)

Phụ thuộc lần train **sau khi gỡ heuristic**. Mốc cũ (eval ~45) là `safe_policy`, **không trích dẫn** là kết quả SARSA.

| Mốc | 10×10 | 30×30 (cùng Q-table) |
|---|---|---|
| Random / greedy Q lúc Q = 0 | ~0–1 | ~0–1 |
| SARSA greedy sau khi ε giảm | đo lại trên `training_log.csv` (`eval_mean_score`) | chơi `python app.py --grid 30` |
| Trần packing tabular | state 7 chiều không thấy hết bàn | cùng giới hạn, bàn rộng hơn |

Chết điển hình: đâm tường/thân, hoặc `no_food` khi đi vòng.

---

## 10. Câu slide gợi ý

1. MDP: agent / env / S / A / R  
2. Vì sao 11.520 state, không encode cả bàn  
3. Safety BFS 4 mức (hình rắn + 3 hướng)  
4. Công thức SARSA vs Q-learning  
5. Bảng reward (shaping ≠ ghi đè action)  
6. Train ε-greedy / eval greedy — cùng Q-table  
7. Bảng mục 8 (vấn đề → giải pháp)  
8. Kết quả 10×10 vs 30×30 và giới hạn tabular  

---

## 11. File code tương ứng

| Chủ đề | File |
|---|---|
| State, BFS, A*, reward, win | `common/snake_env.py` |
| Hệ số reward, lưới 10/30, timeout | `common/env_hyperparameters.py` |
| α, γ, ε, số episode | `agents/sarsa/hyperparameters.py` |
| Q-table SARSA | `agents/sarsa/agent.py` |
| Vòng train + eval greedy | `agents/sarsa/train.py` |
| UI pygame (ô scale theo lưới) | `common/game_window.py`, `app.py` |
