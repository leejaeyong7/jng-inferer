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


class SimultaneousCounter:
    def __init__(self):
        self.state = SimuJumpState.NO_JUMP
        self.count = 0
        self.jumped = True

    def update(self, score):
        type_score = score[:3]
        binary_score = score[3]
        is_jumping = binary_score > 0
        type_ind = np.argmax(type_score)
        if not is_jumping:
            self.state = SimuJumpState.NO_JUMP
            self.jumped = True
        else:
            if type_ind == 1:
                if self.state == SimuJumpState.ROPE_HIGH or self.state == SimuJumpState.NO_JUMP:
                    if self.jumped:
                        self.count += 1
                        self.jumped = False
                self.state = SimuJumpState.JUMP
                self.jumped = True
            elif type_ind == 2:
                self.state = SimuJumpState.ROPE_HIGH
        return self.count


class AlternatingCounter:
    def __init__(self):
        self.state = AltJumpState.NO_JUMP
        self.count = 0
        self.left_jumped = True

    def update(self, score):
        type_score = score[:4]
        binary_score = score[4]
        is_jumping = binary_score > 0
        type_ind = np.argmax(type_score)
        if not is_jumping:
            self.state = AltJumpState.NO_JUMP
            self.left_jumped = True
        else:
            if type_ind == 1:
                if self.state == AltJumpState.ROPE_HIGH or self.state == AltJumpState.NO_JUMP:
                    if self.left_jumped:
                        self.count += 1
                        self.left_jumped = False
                self.state = AltJumpState.RIGHT_JUMP
            elif type_ind == 2:
                self.state = AltJumpState.LEFT_JUMP
                self.left_jumped = True
            elif type_ind == 3:
                self.state = AltJumpState.ROPE_HIGH
        return self.count


def scores_to_counts_simu(scores):
    """
    Args:
        Fx5 score values, where first 4 are type scores, last is binary score
    Returns:
        Fx1 count values
    """
    type_scores = scores[:, :3]
    binary_scores = scores[:, 3]

    counts = []
    counter = SimultaneousCounter()
    for type_score, binary_score in zip(type_scores, binary_scores):
        score = np.concatenate([type_score, [binary_score]])
        counts.append(counter.update(score))
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

    counts = []
    counter = AlternatingCounter()
    for type_score, binary_score in zip(type_scores, binary_scores):
        score = np.concatenate([type_score, [binary_score]])
        counts.append(counter.update(score))
    return counts
