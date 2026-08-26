# Q-Learning Agent (placeholder)

Bạn của bạn thêm code Q-learning vào folder này.

## Gợi ý cấu trúc

```
agents/q_learning/
├── agent.py           # QLearningAgent
├── hyperparameters.py # tham số Q-learning + training
├── train.py           # python -m agents.q_learning.train
└── analyze_games.py   # (tuỳ chọn)
```

## Output model

Lưu vào `output/q_learning/training/`:

- `agent.pkl` — model dùng cho game
- `agent_best.pkl` — checkpoint eval tốt nhất
- `agent_last.pkl` — episode cuối

## Chạy game

```bash
python app.py --agent q_learning
```
