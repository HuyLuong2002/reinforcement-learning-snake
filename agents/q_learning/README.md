# Q-Learning Agent

Agent tabular Q-learning (off-policy) chơi Snake — cấu trúc song song với `agents/sarsa/`.

## Khác SARSA

- **Q-learning:** `Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ]`
- **SARSA:** dùng `Q(s', a')` với `a'` epsilon-greedy thật sự sẽ chọn, không dùng max.

## Cấu trúc

```
agents/q_learning/
├── agent.py           # QLearningAgent
├── hyperparameters.py # tham số Q-learning + training
├── train.py           # python -m agents.q_learning.train
└── analyze_games.py
```

## Chạy

```bash
# Train (chỉnh episodes trong agents/q_learning/hyperparameters.py)
python -m agents.q_learning.train

# Phân tích ván chơi
python -m agents.q_learning.analyze_games --games 10

# Chơi game
python app.py --agent q_learning
```

## Output model

Lưu vào `output/q_learning/training/`:

- `agent.pkl` — model dùng cho game (= best)
- `agent_best.pkl` — checkpoint eval tốt nhất
- `agent_last.pkl` — episode cuối
