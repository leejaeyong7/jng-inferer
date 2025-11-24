import numpy as np

import enum


class JumpState(enum.Enum):
    NO_JUMP = 0
    RIGHT_JUMP = 1
    LEFT_JUMP = 2
    ROPE_HIGH = 3


def scores_to_counts(scores):
    """
    Args:
        Fx5 score values, where first 4 are type scores, last is binary score
    Returns:
        Fx1 count values
    """
    type_scores = scores[
        :, :4
    ]  # 0 = no jump, 1 = right jump, 2 = left jump, 3 = rope high
    binary_scores = scores[:, 4]

    state = JumpState.NO_JUMP
    counts = []

    count = 0
    left_jumped = False
    for type_score, binary_score in zip(type_scores, binary_scores):
        is_jumping = binary_score > 0
        type_ind = np.argmax(type_score)
        if not is_jumping:
            state = JumpState.NO_JUMP
        else:
            if type_ind == 1:
                if state == JumpState.ROPE_HIGH or state == JumpState.NO_JUMP:
                    if left_jumped:
                        count += 1
                        left_jumped = False
                state = JumpState.RIGHT_JUMP
            elif type_ind == 2:
                state = JumpState.LEFT_JUMP
                left_jumped = True
            elif type_ind == 3:
                state = JumpState.ROPE_HIGH

        counts.append(count)
    return counts
