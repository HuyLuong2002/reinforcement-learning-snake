# SARSA / Q-Learning Snake

Đồ án RL: huấn luyện agent tabular SARSA và Q-learning chơi Snake trên lưới.
Đồ án RL: huấn luyện agent tabular SARSA chơi Snake trên lưới 10×10 và 30×30. Cấu trúc mở rộng để bạn thêm Q-learning sau.

Cùng một Q-table chơi được cả hai màn vì state là vector tương đối (không chứa kích thước lưới).

## Cấu trúc

```
├── common/                    # Code dùng chung
│   ├── snake_env.py           # Game Snake (Gymnasium) + BFS + A*
│   ├── game_window.py         # Cửa sổ pygame
│   ├── random_policy.py       # Baseline random
│   ├── training_plots.py      # Biểu đồ learning curve
│   └── env_hyperparameters.py # Tham số môi trường
├── agents/
│   ├── sarsa/                 # SARSA
│   │   ├── agent.py
│   │   ├── hyperparameters.py
│   │   ├── train.py
│   │   └── analyze_games.py
│   └── q_learning/            # Q-learning
│       ├── agent.py
│       ├── hyperparameters.py
│       ├── train.py
│       └── analyze_games.py
├── app.py                     # Mở game (--agent sarsa|q_learning)
└── output/
    ├── sarsa/<YYYY-MM-DD_HH-MM-SS>/   # Mỗi lần train một folder
    └── q_learning/<YYYY-MM-DD_HH-MM-SS>/
```

## Chạy

```bash
pip install -r requirements.txt

# Train (mặc định lưới 15×20)
python -m agents.sarsa.train
python -m agents.q_learning.train --grid 15x20

# Chơi game — menu: tick "Load model", chọn lần train trên select, rồi chọn 10×10 hoặc 30×30
# Bỏ tick = random.
python app.py
python app.py --grid 10
python app.py --grid 30 --random

# Phân tích ván greedy
python -m agents.sarsa.analyze_games --games 10
python -m agents.q_learning.analyze_games --games 10
```

> Dùng `pygame-ce` thay cho `pygame` (Python 3.14). Import vẫn là `import pygame`.

## Policy (SARSA thuần)

| Giai đoạn | Cách chọn action |
| --------- | ---------------- |
| Train     | ε-greedy trên Q-table |
| Eval      | greedy `argmax Q` |
| Chơi      | greedy `argmax Q` |

Phần thưởng A* / giữ đuôi / phạt chết nằm trong `env.step` — đó là tín hiệu học, không phải policy lúc chơi.

## State vector (7 chiều, 11.520 state)

| Feature          | Mô tả                                          |
| ---------------- | ---------------------------------------------- |
| food_dx, food_dy | Hướng thức ăn ∈ {-1,0,1}                       |
| **safety × 3**   | **BFS riêng cho từng hướng** (thẳng/trái/phải) |
| direction        | Hướng rắn                                      |
| fill_bucket      | Mật độ `len(snake) / n²` chia 5 bậc (~20% ô)   |

Mỗi chiều `safety` cho biết đi hướng đó thì chuyện gì xảy ra:

| Giá trị | Ý nghĩa                                                        |
| ------- | -------------------------------------------------------------- |
| 0       | Đâm tường/thân ngay                                            |
| 1       | Số ô tới được < độ dài rắn → không đủ chỗ chứa thân, chết chắc |
| 2       | Đủ chỗ nhưng mất dấu đuôi → rủi ro cao                         |
| 3       | Còn tới được đuôi → an toàn                                    |

**Lưu ý:** Model train bằng state cũ **không tương thích** — phải train lại.

## Checkpoint model

| File | Mô tả |
| ---- | ----- |
| `output/sarsa/<thời-gian>/agent_best.pkl` | Checkpoint eval greedy tốt nhất |
| `output/sarsa/<thời-gian>/agent.pkl` | Model dùng cho game (= best) |
| `output/sarsa/<thời-gian>/agent_last.pkl` | Episode cuối |

Mỗi lần `python -m agents.sarsa.train` tạo folder mới theo thời điểm chạy, không ghi đè lần trước. Trong game, tick **Load model** rồi chọn lần train trên select.

Eval lúc đầu gần random (~0–1) vì Q-table khởi tạo 0. Số ~45 của lần train cũ **không dùng được** — đó là heuristic `safe_policy`, không phải SARSA.

## Kết thúc một ván

| `end_reason` | Nghĩa                                                          |
| ------------ | -------------------------------------------------------------- |
| `death`      | Va tường hoặc va thân                                          |
| `max_steps`  | Hết trần bước (10×10: 2000; 30×30 scale theo diện tích)        |
| `no_food`    | Đi `max_steps_without_food` bước liên tiếp mà không ăn được gì |
| `win`        | Fill hết bàn                                                   |

## Q-learning vs SARSA

Q-learning (off-policy) cập nhật bằng `max_a' Q(s', a')`. SARSA (on-policy) dùng `Q(s', a')` với action epsilon-greedy thật sự sẽ chọn. Hyperparameter mặc định giống nhau để so sánh công bằng. Xem `agents/q_learning/README.md`.
