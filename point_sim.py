"""Map processing and reinforcement-learning model for the virtual fly."""

import math
import random
from array import array
from typing import Any

from PIL import Image


GRID_STEP = 4
ACTION_COUNT = 8
GOAL_DIRECTION_COUNT = 8
ACTION_MASK_COUNT = 1 << ACTION_COUNT
POSITION_BINS_X = 16
POSITION_BINS_Y = 8
LOCAL_STATE_COUNT = GOAL_DIRECTION_COUNT * ACTION_MASK_COUNT
STATE_COUNT = POSITION_BINS_X * POSITION_BINS_Y * LOCAL_STATE_COUNT
MAX_TRIAL_STEPS = 2500
TRAINING_EPISODES_PER_CLICK = 20
LEARNING_RATE = 0.18
DISCOUNT_FACTOR = 0.995
MOVE_DIRECTIONS = (
    (-1, -1), (0, -1), (1, -1), (-1, 0),
    (1, 0), (-1, 1), (0, 1), (1, 1),
)


def create_walkable_mask(image: Image.Image) -> tuple[int, int, bytes]:
    """Create a packed walkable-cell mask from the map's gray corridors."""
    width, height = image.size
    grid_width = (width + GRID_STEP - 1) // GRID_STEP
    grid_height = (height + GRID_STEP - 1) // GRID_STEP
    sampled = image.resize(
        (grid_width, grid_height),
        Image.Resampling.BILINEAR,
    ).convert("RGB")
    packed = bytearray((grid_width * grid_height + 7) // 8)
    walkable_count = 0

    index = 0
    for y in range(grid_height):
        for x in range(grid_width):
            red, green, blue = sampled.getpixel((x, y))
            brightness = (red + green + blue) // 3
            saturation = max(red, green, blue) - min(red, green, blue)
            if 35 <= brightness <= 125 and saturation <= 35:
                packed[index >> 3] |= 1 << (index & 7)
                walkable_count += 1
            index += 1

    if walkable_count == 0:
        raise ValueError("회색 통로 후보를 찾지 못했습니다. 지도 색상 기준을 확인하세요.")
    return grid_width, grid_height, bytes(packed)


class FlyExperiment:
    """Owns map navigation, circuit influence, and Q-learning state."""

    def __init__(
        self,
        image_size: tuple[int, int],
        mask_size: tuple[int, int],
        walkable_bits: bytes,
        circuit_data: dict[str, Any],
    ) -> None:
        self.image_width, self.image_height = image_size
        self.mask_width, self.mask_height = mask_size
        self.walkable_bits = walkable_bits
        self.roles = circuit_data["roles"]
        self.stats = circuit_data["stats"]
        self.profiles = self._build_circuit_profiles(circuit_data["visual_edges"])
        self.q_values = array("f", [0.0]) * (STATE_COUNT * ACTION_COUNT)
        self.training_episodes = 0
        self.successful_training_episodes = 0
        self.start: tuple[float, float] | None = None
        self.destination: tuple[float, float] | None = None
        self.point: tuple[float, float] | None = None
        self.path: list[tuple[float, float]] = []
        self.trial_step_count = 0
        self.is_running = False
        self.message = "지도를 클릭해 시작점과 도착점을 순서대로 지정하세요."

    def _build_circuit_profiles(
        self, edges: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        profiles = {
            role["root_id"]: {
                **role,
                "left": 0,
                "right": 0,
                "total": 0,
            }
            for role in self.roles
        }
        for edge in edges:
            profile = profiles.get(edge["post_id"])
            if profile is None:
                continue
            synapse_count = edge["synapse_count"]
            profile["total"] += synapse_count
            if edge["pre_side"] == "left":
                profile["left"] += synapse_count
            elif edge["pre_side"] == "right":
                profile["right"] += synapse_count
        return list(profiles.values())

    def _is_walkable(self, x: float, y: float) -> bool:
        if not (0 <= x < self.image_width and 0 <= y < self.image_height):
            return False
        grid_x = min(self.mask_width - 1, int(x // GRID_STEP))
        grid_y = min(self.mask_height - 1, int(y // GRID_STEP))
        index = grid_y * self.mask_width + grid_x
        return bool(self.walkable_bits[index >> 3] & (1 << (index & 7)))

    def _to_grid_point(self, grid_x: int, grid_y: int) -> tuple[float, float]:
        return (
            min(self.image_width - 1, grid_x * GRID_STEP + GRID_STEP / 2),
            min(self.image_height - 1, grid_y * GRID_STEP + GRID_STEP / 2),
        )

    def _is_at_goal(
        self, position: tuple[float, float], goal: tuple[float, float]
    ) -> bool:
        return (
            int(position[0] // GRID_STEP) == int(goal[0] // GRID_STEP)
            and int(position[1] // GRID_STEP) == int(goal[1] // GRID_STEP)
        )

    def _legal_actions(self, position: tuple[float, float]) -> list[int]:
        x = int(position[0] // GRID_STEP)
        y = int(position[1] // GRID_STEP)
        actions = []
        for action, (dx, dy) in enumerate(MOVE_DIRECTIONS):
            next_x = x + dx
            next_y = y + dy
            if not (0 <= next_x < self.mask_width and 0 <= next_y < self.mask_height):
                continue
            next_point = self._to_grid_point(next_x, next_y)
            if not self._is_walkable(*next_point):
                continue
            if dx and dy:
                horizontal = self._to_grid_point(next_x, y)
                vertical = self._to_grid_point(x, next_y)
                if not self._is_walkable(*horizontal) or not self._is_walkable(*vertical):
                    continue
            actions.append(action)
        return actions

    def _state(
        self,
        position: tuple[float, float],
        goal: tuple[float, float],
        actions: list[int],
    ) -> int:
        angle = math.atan2(goal[1] - position[1], goal[0] - position[0])
        goal_direction = (math.floor(angle / (math.pi / 4) + 0.5) + 8) % 8
        action_mask = sum(1 << action for action in actions)
        local_state = goal_direction * ACTION_MASK_COUNT + action_mask
        x = int(position[0] // GRID_STEP)
        y = int(position[1] // GRID_STEP)
        bin_x = min(POSITION_BINS_X - 1, x * POSITION_BINS_X // self.mask_width)
        bin_y = min(POSITION_BINS_Y - 1, y * POSITION_BINS_Y // self.mask_height)
        position_bin = bin_y * POSITION_BINS_X + bin_x
        return position_bin * LOCAL_STATE_COUNT + local_state

    def _circuit_biases(
        self, position: tuple[float, float], goal: tuple[float, float]
    ) -> list[float]:
        dx = goal[0] - position[0]
        dy = goal[1] - position[1]
        distance = math.hypot(dx, dy) or 1
        horizontal_cue = dx / distance
        left_cue = max(0.0, -horizontal_cue)
        right_cue = max(0.0, horizontal_cue)

        steering_left = 0.0
        steering_right = 0.0
        forward_total = 0.0
        forward_count = 0
        for node in self.profiles:
            total = node["total"]
            activity = (
                (
                    0.2 * total
                    + 0.8 * (
                        node["left"] * left_cue + node["right"] * right_cue
                    )
                )
                / total
                if total
                else 0.0
            )
            if node["model_role"] == "steering" and node["side"] == "left":
                steering_left = activity
            elif node["model_role"] == "steering" and node["side"] == "right":
                steering_right = activity
            elif node["model_role"] == "forward":
                forward_total += activity
                forward_count += 1

        forward = forward_total / max(1, forward_count)
        goal_dx = (goal[0] > position[0]) - (goal[0] < position[0])
        return [
            0.08 * (steering_right - steering_left) * move_dx * goal_dx
            + 0.04 * forward * (move_dx * goal_dx)
            for move_dx, _ in MOVE_DIRECTIONS
        ]

    def _select_action(
        self,
        state: int,
        actions: list[int],
        position: tuple[float, float],
        goal: tuple[float, float],
        explore: float,
    ) -> int:
        if explore and random.random() < explore:
            return random.choice(actions)
        biases = self._circuit_biases(position, goal)
        return max(
            actions,
            key=lambda action: self.q_values[state * ACTION_COUNT + action]
            + biases[action],
        )

    def _step_to(self, position: tuple[float, float], action: int) -> tuple[float, float]:
        x = int(position[0] // GRID_STEP)
        y = int(position[1] // GRID_STEP)
        dx, dy = MOVE_DIRECTIONS[action]
        return self._to_grid_point(x + dx, y + dy)

    def select_point(self, x: Any, y: Any) -> None:
        if (
            isinstance(x, bool)
            or isinstance(y, bool)
            or not isinstance(x, (int, float))
            or not isinstance(y, (int, float))
            or (isinstance(x, float) and not math.isfinite(x))
            or (isinstance(y, float) and not math.isfinite(y))
        ):
            raise ValueError("지도 위치는 유효한 숫자여야 합니다.")
        if not self._is_walkable(x, y):
            raise ValueError("짙은 회색 통로 안쪽을 선택하세요.")

        grid_x = int(x // GRID_STEP)
        grid_y = int(y // GRID_STEP)
        selected = self._to_grid_point(grid_x, grid_y)
        self.is_running = False
        self.trial_step_count = 0
        if self.start is None or self.destination is not None:
            self.start = selected
            self.destination = None
            self.point = selected
            self.path = []
            self.message = "시작점을 지정했습니다. 끝점을 선택하세요."
        else:
            self.destination = selected
            self.point = self.start
            self.path = [self.start]
            self.message = (
                "끝점을 지정했습니다. 먼저 학습하세요."
                if self.successful_training_episodes == 0
                else "끝점을 지정했습니다. 학습 또는 실행을 선택하세요."
            )

    def train(self) -> None:
        if self.start is None or self.destination is None:
            raise ValueError("학습 전에 시작점과 끝점을 지정하세요.")

        successes = 0
        for episode in range(TRAINING_EPISODES_PER_CLICK):
            position = self.start
            if self._is_at_goal(position, self.destination):
                successes += 1
                continue
            for step in range(MAX_TRIAL_STEPS):
                actions = self._legal_actions(position)
                if not actions:
                    break
                state = self._state(position, self.destination, actions)
                epsilon = max(
                    0.05,
                    0.7 * math.exp(-(self.training_episodes + episode) / 80),
                )
                action = self._select_action(
                    state, actions, position, self.destination, epsilon
                )
                next_position = self._step_to(position, action)
                distance_before = math.dist(position, self.destination)
                distance_after = math.dist(next_position, self.destination)
                reached = self._is_at_goal(next_position, self.destination)
                speed_bonus = (
                    20 * (MAX_TRIAL_STEPS - step) / MAX_TRIAL_STEPS if reached else 0
                )
                next_actions = (
                    [] if reached else self._legal_actions(next_position)
                )
                trapped = not reached and not next_actions
                reward = (
                    -0.05
                    + 0.1 * (distance_before - distance_after) / GRID_STEP
                    - (3 if trapped else 0)
                    + (30 + speed_bonus if reached else 0)
                )
                next_state = (
                    self._state(next_position, self.destination, next_actions)
                    if next_actions
                    else 0
                )
                best_next_value = max(
                    (
                        self.q_values[next_state * ACTION_COUNT + next_action]
                        for next_action in next_actions
                    ),
                    default=0.0,
                )
                q_index = state * ACTION_COUNT + action
                target = reward + (
                    0.0
                    if reached
                    else DISCOUNT_FACTOR * best_next_value
                )
                self.q_values[q_index] += LEARNING_RATE * (
                    target - self.q_values[q_index]
                )
                position = next_position
                if reached:
                    successes += 1
                    break

        self.training_episodes += TRAINING_EPISODES_PER_CLICK
        self.successful_training_episodes += successes
        self.message = (
            f"학습 완료: {successes}/{TRAINING_EPISODES_PER_CLICK}회 도착 · "
            f"총 {self.training_episodes}회 학습"
        )

    def start_run(self) -> None:
        if self.start is None or self.destination is None:
            raise ValueError("실행 전에 시작점과 끝점을 지정하세요.")
        if self.successful_training_episodes == 0:
            raise ValueError("먼저 학습 버튼을 눌러 도착에 성공한 기록을 만드세요.")
        self.point = self.start
        self.path = [self.start]
        self.trial_step_count = 0
        if self._is_at_goal(self.start, self.destination):
            self.is_running = False
            self.message = "시작점이 이미 목표 위치입니다."
            return
        self.is_running = True
        self.message = "실험 실행 중: Python 모델이 매 스텝의 이동을 선택합니다."

    def step(self) -> None:
        if not self.is_running or self.point is None or self.destination is None:
            return
        actions = self._legal_actions(self.point)
        if not actions:
            self.is_running = False
            self.message = f"실험 실패: 이동 가능한 방향이 없습니다 ({self.trial_step_count}스텝)."
            return
        state = self._state(self.point, self.destination, actions)
        action = self._select_action(
            state, actions, self.point, self.destination, explore=0
        )
        self.point = self._step_to(self.point, action)
        self.path.append(self.point)
        self.trial_step_count += 1
        if self._is_at_goal(self.point, self.destination):
            self.is_running = False
            self.message = f"실험 성공: 목표에 도착했습니다 ({self.trial_step_count}스텝)."
        elif self.trial_step_count >= MAX_TRIAL_STEPS:
            self.is_running = False
            self.message = (
                f"실험 실패: {MAX_TRIAL_STEPS}스텝 안에 목표에 도착하지 못했습니다."
            )

    def reset_learning(self) -> None:
        self.q_values = array("f", [0.0]) * (STATE_COUNT * ACTION_COUNT)
        self.training_episodes = 0
        self.successful_training_episodes = 0
        self.is_running = False
        self.message = "학습 기록을 초기화했습니다."

    def reset_points(self) -> None:
        self.start = None
        self.destination = None
        self.point = None
        self.path = []
        self.trial_step_count = 0
        self.is_running = False
        self.message = "지도를 클릭해 시작점을 지정하세요."

    def public_state(self, include_path: bool = True) -> dict[str, Any]:
        state = {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "grid_step": GRID_STEP,
            "stats": self.stats,
            "start": self._point_json(self.start),
            "destination": self._point_json(self.destination),
            "point": self._point_json(self.point),
            "trial_step_count": self.trial_step_count,
            "training_episodes": self.training_episodes,
            "successful_training_episodes": self.successful_training_episodes,
            "is_running": self.is_running,
            "message": self.message,
        }
        if include_path:
            state["path"] = [self._point_json(point) for point in self.path]
        return state

    @staticmethod
    def _point_json(point: tuple[float, float] | None) -> dict[str, float] | None:
        if point is None:
            return None
        return {"x": point[0], "y": point[1]}
