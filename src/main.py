from backend.scene_detect import *
from plain_player import *
import subprocess
import os   
from create_trajectory_imgs import track_and_draw_on_first_frame
# Parse command line arguments
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Split video into segments based on camera movements')
    parser.add_argument('--video', '-v', type=str, required=False, 
                        default=os.path.join(os.getcwd(), "videos", "alabama_clemson.mp4"),
                        help='Path to the input video file')
                  
    args = parser.parse_args()
    print(f"Input video: {args.video}")

    # Run the processing function
    segments = split_camera_moves(
        args.video, 
        output_dir=os.path.join(os.getcwd(), "video_segments")
    )

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

    # run the plain player.py file using subprocess
    new_json = []
    for segment in segments:
        position_json = get_player_positions(segment)
        player_numbers = unique_jersey_number(position_json) #unique jersey numbers
        temp_json = {
            "video_segment_path": segment,
            "frame_list": position_json,   #frame list stores a list of player positions for frame range
        }
#{
#            "jerseyNumber": 10,
#            "coordinates": { "x_coordinate": 0.139, "y_coordinate": 0.526 },
#            "team": "Clemson"
#          },
        for frame_range in temp_json["frame_list"]:
            s = convert_frame_to_seconds(frame_range["start"])
            e = convert_frame_to_seconds(frame_range["end"])
            for img in frame_range["image_data"]["players"]:
                x, y = convert_percent_to_coordinates(
                    img["coordinates"]["x_coordinate"],
                    img["coordinates"]["y_coordinate"],
                    segment
                )
                print("Processing player:", img["jerseyNumber"], "at coordinates:", x, y)
                track_and_draw_on_first_frame(
                    video_path=segment,
                    start_time=s, 
                    end_time=e + 2 if e+2 < get_video_seconds(segment) else get_video_seconds(segment),
                    cx=x, 
                    cy=y,
                    output_filename=os.path.join(os.getcwd(), "trajectory_images", f"{img['jerseyNumber']}_{img['team']}.png")
                    )
                
            
        break
        




    

        

       

        




    





