"""Training-only reflection about the tag-10 X/Z plane for single-drone room data.

The room/tag layout is not exactly symmetric. These transforms provide imitation
examples, not fabricated on-policy transitions or a runtime navigation controller.
Candidates must be evaluated in the actual, unmodified room.
"""


def mirror_room_observation(observation):
    if observation.shape[-1] % 25:
        raise ValueError("Expected 25-field room observations or stacked histories")
    result = observation.clone() if hasattr(observation, "clone") else observation.copy()
    frames = result.reshape(*result.shape[:-1], -1, 25)
    # World Y, sin(yaw), rightward goal offsets, sin(tag bearing), previous right/yaw RC.
    frames[..., [1, 4, 9, 12, 15, 18, 21]] *= -1
    return result


def mirror_room_action(action):
    if action.shape[-1] != 5:
        raise ValueError("Expected [right, forward, up, clockwise yaw, land]")
    result = action.clone() if hasattr(action, "clone") else action.copy()
    result[..., [0, 3]] *= -1
    return result
