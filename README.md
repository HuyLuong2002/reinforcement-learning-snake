# SARSA / Q-Learning Snake

Đồ án RL: huấn luyện agent tabular SARSA và Q-learning chơi Snake trên lưới.

## Cấu trúc

```
├── common/                    # Code dùng chung
│   ├── snake_env.py           # Game Snake (Gymnasium) + BFS reachable
│   ├── game_window.py         # Cửa sổ pygame
│   ├── random_policy.py       # Baseline random
│   ├── training_plots.py      # Biểu đồ learning curve
│   └── env_hyperparameters.py # Tham số môi trường
├── agents/
│   ├── sarsa/                 # SARSA của bạn
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
    ├── sarsa/training/        # Model SARSA
    └── q_learning/training/   # Model Q-learning
```

## Chạy

```bash
pip install -r requirements.txt

# Train SARSA (chỉnh episodes trong agents/sarsa/hyperparameters.py)
python -m agents.sarsa.train

# Train Q-learning
python -m agents.q_learning.train

# Chơi game
python app.py
python app.py --agent sarsa
python app.py --agent q_learning

# Phân tích ván chơi
python -m agents.sarsa.analyze_games --games 10
python -m agents.q_learning.analyze_games --games 10
```

> Dùng `pygame-ce` thay cho `pygame` (Python 3.14). Import vẫn là `import pygame`.

## State vector (7 chiều, 11.520 state)

| Feature          | Mô tả                                          |
| ---------------- | ---------------------------------------------- |
| food_dx, food_dy | Hướng thức ăn ∈ {-1,0,1}                       |
| **safety × 3**   | **BFS riêng cho từng hướng** (thẳng/trái/phải) |
| direction        | Hướng rắn                                      |
| length_bucket    | Score // 5 (0–4)                               |

Mỗi chiều `safety` cho biết đi hướng đó thì chuyện gì xảy ra:

| Giá trị | Ý nghĩa                                                       |
| ------- | ------------------------------------------------------------- |
| 0       | Đâm tường/thân ngay                                            |
| 1       | Số ô tới được < độ dài rắn → không đủ chỗ chứa thân, chết chắc |
| 2       | Đủ chỗ nhưng mất dấu đuôi → rủi ro cao                         |
| 3       | Còn tới được đuôi → an toàn                                    |

Tiêu chí "có tới được đuôi không" là mấu chốt: nếu đầu rắn còn đường tới đuôi thì
nó luôn có thể bám theo đuôi mà sống tiếp. Tiêu chí này không phụ thuộc độ dài rắn
nên vẫn có ý nghĩa cả khi rắn đã rất dài.

**Lưu ý:** Model train bằng state cũ **không tương thích** — phải train lại.

## Checkpoint model

Cùng tên file, khác thư mục (`output/sarsa/training/` hoặc `output/q_learning/training/`):

| File           | Mô tả                        |
| -------------- | ---------------------------- |
| `agent_best.pkl` | Checkpoint eval tốt nhất     |
| `agent.pkl`      | Model dùng cho game (= best) |
| `agent_last.pkl` | Episode cuối                 |

## Kết thúc một ván

| `end_reason` | Nghĩa |
|--------------|-------|
| `death` | Va tường hoặc va thân |
| `max_steps` | Hết trần bước (`max_steps`, mặc định 2000) |
| `no_food` | Đi `max_steps_without_food` bước liên tiếp mà không ăn được gì |

## Q-learning vs SARSA

Q-learning (off-policy) cập nhật bằng `max_a' Q(s', a')`. SARSA (on-policy) dùng `Q(s', a')` với action epsilon-greedy thật sự sẽ chọn. Hyperparameter mặc định giống nhau để so sánh công bằng. Xem `agents/q_learning/README.md`.
