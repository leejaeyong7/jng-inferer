import numpy as np

import enum


class SimuJumpState(enum.Enum):
    NO_JUMP = 0
    JUMP = 1
    ROPE_HIGH = 2


class AltJumpState(enum.Enum):
    NO_JUMP = 0
    RIGHT_JUMP = 1
    LEFT_JUMP = 2
    ROPE_HIGH = 3


def scores_to_counts_simu(scores):
    """
    Args:
        Fx5 score values, where first 4 are type scores, last is binary score
    Returns:
        Fx1 count values
    """
    type_scores = scores[:, :3]
    binary_scores = scores[:, 3]

    state = SimuJumpState.NO_JUMP
    counts = []

    count = 0
    jumped = True
    for type_score, binary_score in zip(type_scores, binary_scores):
        is_jumping = binary_score > 0
        type_ind = np.argmax(type_score)
        if not is_jumping:
            state = SimuJumpState.NO_JUMP
            jumped = True
        else:
            if type_ind == 1:
                if state == SimuJumpState.ROPE_HIGH or state == SimuJumpState.NO_JUMP:
                    if jumped:
                        count += 1
                        jumped = False
                state = SimuJumpState.JUMP
                jumped = True
            elif type_ind == 2:
                state = SimuJumpState.ROPE_HIGH

        counts.append(count)
    return counts


def scores_to_counts_alt(scores):
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

    state = AltJumpState.NO_JUMP
    counts = []

    count = 0
    left_jumped = True
    for type_score, binary_score in zip(type_scores, binary_scores):
        is_jumping = binary_score > 0
        type_ind = np.argmax(type_score)
        if not is_jumping:
            state = AltJumpState.NO_JUMP
            left_jumped = True
        else:
            if type_ind == 1:
                if state == AltJumpState.ROPE_HIGH or state == AltJumpState.NO_JUMP:
                    if left_jumped:
                        count += 1
                        left_jumped = False
                state = AltJumpState.RIGHT_JUMP
            elif type_ind == 2:
                state = AltJumpState.LEFT_JUMP
                left_jumped = True
            elif type_ind == 3:
                state = AltJumpState.ROPE_HIGH

        counts.append(count)
    return counts
