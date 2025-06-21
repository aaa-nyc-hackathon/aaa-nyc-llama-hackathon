import cv2 as cv

# Fixed player target box
player_box_width = 30
player_box_height = 60


# Add box around target points
def highlight_box(
    frame: cv.typing.MatLike, pX, pY, color=(0, 255, 0)
) -> cv.typing.MatLike:
    rect_start = (
        int(pX) - (player_box_width // 2),
        int(pY) - (player_box_height // 3 * 2),
    )
    rect_end = (int(pX) + (player_box_width // 3), int(pY) + (player_box_height // 2))

    return cv.rectangle(frame, rect_start, rect_end, color, thickness=2)
