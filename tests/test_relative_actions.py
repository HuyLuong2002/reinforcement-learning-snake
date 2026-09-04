"""Action space: 3 hướng hợp lệ (thẳng / trái / phải), không còn lùi 180°."""

from __future__ import annotations

import unittest

from common.snake_env import (
    DOWN,
    LEFT,
    LEFT_TURN,
    REL_LEFT,
    REL_RIGHT,
    REL_STRAIGHT,
    RIGHT,
    RIGHT_TURN,
    UP,
    SnakeEnv,
    relative_to_absolute,
)


class RelativeActionSpaceTests(unittest.TestCase):
    def test_action_space_has_three_actions(self) -> None:
        env = SnakeEnv(width=10, height=10)
        self.assertEqual(env.n_actions, 3)
        self.assertEqual(int(env.action_space.n), 3)

    def test_relative_maps_to_heading(self) -> None:
        self.assertEqual(relative_to_absolute(RIGHT, REL_STRAIGHT), RIGHT)
        self.assertEqual(relative_to_absolute(RIGHT, REL_LEFT), UP)
        self.assertEqual(relative_to_absolute(RIGHT, REL_RIGHT), DOWN)
        self.assertEqual(relative_to_absolute(UP, REL_LEFT), LEFT)
        self.assertEqual(relative_to_absolute(UP, REL_RIGHT), RIGHT)

    def test_step_straight_keeps_heading(self) -> None:
        env = SnakeEnv(width=10, height=10)
        env.reset(seed=0)
        assert env.state is not None
        heading = env.state.direction
        env.step(REL_STRAIGHT)
        assert env.state is not None
        self.assertEqual(env.state.direction, heading)

    def test_step_left_turns_relative(self) -> None:
        env = SnakeEnv(width=10, height=10)
        env.reset(seed=0)
        assert env.state is not None
        heading = env.state.direction
        env.step(REL_LEFT)
        assert env.state is not None
        self.assertEqual(env.state.direction, LEFT_TURN[heading])

    def test_step_right_turns_relative(self) -> None:
        env = SnakeEnv(width=10, height=10)
        env.reset(seed=0)
        assert env.state is not None
        heading = env.state.direction
        env.step(REL_RIGHT)
        assert env.state is not None
        self.assertEqual(env.state.direction, RIGHT_TURN[heading])

    def test_cannot_reverse_in_one_step(self) -> None:
        env = SnakeEnv(width=10, height=10)
        env.reset(seed=0)
        assert env.state is not None
        start = env.state.direction
        for action in (REL_STRAIGHT, REL_LEFT, REL_RIGHT):
            env.reset(seed=0)
            env.step(action)
            assert env.state is not None
            self.assertNotEqual(env.state.direction, {UP: DOWN, RIGHT: LEFT, DOWN: UP, LEFT: RIGHT}[start])


if __name__ == "__main__":
    unittest.main()
