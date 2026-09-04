# Q-Learning Agent

Agent tabular Q-learning (**off-policy**) chơi Snake — cùng env / `select_action` / hyperparameter với SARSA.

## Policy

- **SARSA = on-policy:** cập nhật bằng `Q(s', a')` với `a'` epsilon-greedy thật sự sẽ chọn.
- **Q-learning = off-policy:** cập nhật bằng `max Q(s', a')` trên 3 hướng hợp lệ (cấm 180°).

Cả hai lúc train/eval/chơi đều chọn action qua `common.policy.select_action` (ε-greedy / greedy, cấm 180°). Không heuristic, không A*.

## Chạy

```bash
# Train mặc định 15×20, 30000 episode (cùng SARSA)
python -m agents.q_learning.train

python -m agents.q_learning.analyze_games --games 10
python app.py --agent q_learning
```

Output: `output/q_learning/YYYY-MM-DD_HH-MM-SS/` (`agent.pkl` = best).
