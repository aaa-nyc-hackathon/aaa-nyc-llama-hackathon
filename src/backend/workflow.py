from scene_detect import *
from plain_player import *
import os
import time
import json
from collections import defaultdict
from create_trajectory_imgs import track_and_draw_on_first_frame
from player_feedback import get_player_feedback
import shutil
# Parse command line arguments



def extract_team_data(frame_list):
    team_data = defaultdict(lambda: defaultdict(list))

    for frame in frame_list:
        start = frame["start"]
        end = frame["end"]
        team_1_name = frame["image_data"]["teamName"]
        team_2_name = frame["image_data"]["teamName"]
        team_1_score = frame["image_data"]["score"]
        team_2_score = frame["image_data"]["score"]
        players = frame["image_data"]["players"]

        for player in players:
            print(player.keys())
            print(player["coordinates"].keys())
            team = player["team"]
            jersey_number = player["jerseyNumber"]
            x = player["coordinates"]["x_coordinate"]
            y = player["coordinates"]["y_coordinate"]

            # Append the player's data to the respective team
            team_data[team][jersey_number].append({
                "start_frame": start,
                "end_frame": end,
                "x": x,
                "y": y,
                "player_team_score": team_1_score if team == team_1_name else team_2_score,
                "opponent_team_score": team_2_score if team == team_1_name else team_1_score
            })

    # Transform the defaultdict into the desired format
    result = []
    for team, players in team_data.items():
        player_objects = []
        for jersey_number, data in players.items():
            player_objects.append({
                "jersey_number": jersey_number,
                "frames": data
            })
        result.append({"team": team, "obj": player_objects})

    return result

def unique_jersey_number(positions):
    """
    Extract unique player numbers from the positions list.
    """
    player_numbers = set()
    for position in positions:
        if "jerseyNumber" in position:
            player_numbers.add((position["jerseyNumber"], position["team"]))
    return list(player_numbers)

def get_video_frame_size(video_path):
    """
    Get the size of the video frames.
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height

def get_video_seconds(video_path):
    """
    Get the total duration of the video in seconds.
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()
    return duration

def convert_percent_to_coordinates(percent_x, percent_y, video_path):
    """
    Convert percentage coordinates to pixel coordinates based on video frame size.
    """
    width, height = get_video_frame_size(video_path)
    x_coordinate = int(percent_x * width)
    y_coordinate = int(percent_y * height)
    return x_coordinate, y_coordinate

def convert_frame_to_seconds(frame_number, fps=1):
    """
    Convert frame number to seconds based on the video FPS.
    """
    if fps <= 0:
        return 0.0
    return frame_number / fps

def start_workflow(source_video_path):
    segments = split_camera_moves(source_video_path)
    new_json = []
    clip_number = 0
    for segment in segments:
        time.sleep(30)
        print("sleeping for 30 seconds")
        try:
            position_json = get_player_positions(segment)
        except Exception as e:
            print(f"Error getting player positions: {e} retrying since probably a structured output error")
            position_json = get_player_positions(segment)
        player_numbers = unique_jersey_number(position_json) #unique jersey numbers
        temp_json = {
            "video_segment_path": segment,
            "frame_list": position_json,   #frame list stores a list of player positions for frame range
        }

        json_team_player_timeframe = extract_team_data(temp_json["frame_list"])
        if os.path.exists(os.path.join(os.getcwd(), "trajectory_images")):
            shutil.rmtree(os.path.join(os.getcwd(), "trajectory_images"))


        for team in json_team_player_timeframe:
            team_name = team["team"]
            for player in team["obj"]:
                jersey_number = player["jersey_number"]
                frames = player["frames"]
                for frame in frames:
                    s = frame["start_frame"]
                    e = frame["end_frame"]
                    x, y = convert_percent_to_coordinates(frame["x"], frame["y"], segment)
                    print("Processing player:", jersey_number, "at coordinates:", x, y)
                    marked_up_img_path = track_and_draw_on_first_frame(
                        video_path=segment,
                        start_time=s,
                        end_time=e + 2 if e+2 < get_video_seconds(segment) else get_video_seconds(segment),
                        cx=x,
                        cy=y,
                        output_filename=os.path.join(os.getcwd(), "trajectory_images", f"{jersey_number}_{team_name}_start_{s}_end_{e}.png")
                    )
                    # add the marked up image path to the json
                    frame["marked_up_image_path"] = marked_up_img_path






        for team in json_team_player_timeframe:
            for player in team["obj"]:
                jersey_number = player["jersey_number"]
                #########big edit to distill data
                frame_list = player["frames"] if len(player["frames"]) <3 else player["frames"][:2]  # Limit to first 2 frames for feedback
                for frame in player["frames"]:
                    if frame["marked_up_image_path"] is None:
                        print(f"Marked up image path is None for player {jersey_number} in team {team['team']}")
                        continue
                    if not os.path.exists(frame["marked_up_image_path"]):
                        print(f"Marked up image path does not exist: {frame['marked_up_image_path']}")
                        continue
                    try:
                        feedback = get_player_feedback(frame["marked_up_image_path"])
                        frame["feedback"] = feedback
                    except Exception as e:
                        frame["feedback"] = None


        output = {
            "video_segment_path": segment,
            "list_of_info": json_team_player_timeframe
        }
        print(json.dumps(output, indent=2))



        clip_number += 1

        new_json.append(output)
        break
    outFinal = {"video_segments": new_json}
    try:
        with open(os.path.join(os.getcwd(), "final_output.json"), "w") as f:
            json.dump(outFinal, f, indent=2)
        return outFinal
    except Exception as e:
        print(f"Error writing final output: {e}")
        return None





















