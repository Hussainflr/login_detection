import cv2
from skimage.metrics import structural_similarity as ssim
from config import OUTPUT_VIDEO_PATH, CODEC_FORMAT
import cv2
from skimage.metrics import structural_similarity as ssim
from config import  MODELS, OUTPUT_VIDEO_PATH, SIMILARITY_THRESHOLD, CODEC_FORMAT
from utility import  format_data
from model import getmodel


def compare_orb(frame1, frame2):
    orb = cv2.ORB_create()
    
    # Detect ORB keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(frame1, None)
    kp2, des2 = orb.detectAndCompute(frame2, None)
    
    # Use BFMatcher to match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Calculate match ratio
    match_ratio = len(matches) / max(len(kp1), len(kp2))
    return match_ratio


# Function to compute SSIM between two frames
def calculate_ssim(frame1, frame2):
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(frame1_gray, frame2_gray, full=True)
    return score

# Function to remove similar frames and return processed video
def remove_similar_frames(input_video, similarity_threshold=0.7):
    # Get video properties from the input
    
    input_video = cv2.VideoCapture(input_video)
    frame_rate = input_video.get(cv2.CAP_PROP_FPS)
    width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create VideoWriter object to output the processed video
    fourcc = cv2.VideoWriter_fourcc(*CODEC_FORMAT)
    out_video_path = OUTPUT_VIDEO_PATH
    out = cv2.VideoWriter(out_video_path, fourcc, frame_rate, (width, height))

    ret, prev_frame = input_video.read()
    if not ret:
        print("Failed to read video")
        return None  # Return None if the video can't be read

    out.write(prev_frame)  # Write the first frame to the output
    count = 0
    while True:
        ret, current_frame = input_video.read()
        if not ret:
            break  # End of video

        # Calculate SSIM between current frame and previous frame
        similarity = calculate_ssim(prev_frame, current_frame)
        # similarity = compare_orb(prev_frame, current_frame)
        if similarity < similarity_threshold:  # If frames are not similar, write to output
            out.write(current_frame)
            count +=1
            prev_frame = current_frame  # Update previous frame

    # Release the output video writer object
    out.release()

    # Return the processed video object
    processed_video = cv2.VideoCapture(OUTPUT_VIDEO_PATH)
    print(">>>>>>>>>>  ", processed_video)
    return processed_video, out_video_path




def extract_login_frames(input_video, modelname, similarity_threshold=SIMILARITY_THRESHOLD):
  model = getmodel(modelname)
  input_video = cv2.VideoCapture(input_video)
  frame_rate = input_video.get(cv2.CAP_PROP_FPS)
  width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
  totalframes = input_video.get(cv2.CAP_PROP_FRAME_COUNT)

  # Create VideoWriter object to output the processed video
  fourcc = cv2.VideoWriter_fourcc(*CODEC_FORMAT)
  out_video_path = OUTPUT_VIDEO_PATH
  out = cv2.VideoWriter(out_video_path, fourcc, frame_rate, (width, height))
  login_probs = []
  nonlogin_probs = []
  lframes = 0 # login frames count
  nlframes = 0 # nonlogin frames count
  dframes = 0 # login duplicate frames count

  ret, prev_frame = input_video.read()
  if not ret:
      print("Failed to read video")
      return None  # Return None if the video can't be read

  out.write(prev_frame)  # Write the first frame to the output
  count = 0
  while True:
    ret, current_frame = input_video.read()
    if not ret:
        break  # End of video
    results = model(current_frame)
    probs = results[0].probs.data.numpy()
    login_probs.append(probs[0])
    nonlogin_probs.append(probs[1])
    if probs[0] > 0.8:
      # Calculate SSIM between current frame and previous frame
      similarity = calculate_ssim(prev_frame, current_frame)
      # similarity = compare_orb(prev_frame, current_frame)
      if similarity < similarity_threshold:  # If frames are not similar, write to output
          out.write(current_frame)
          count +=1
          prev_frame = current_frame  # Update previous frame
          lframes +=1
      else:
         dframes +=1
    else:
       nlframes +=1
  out.release()
  # Return the processed video object
  login_avg = sum(login_probs)/len(login_probs)
  nonlogin_avg = sum(nonlogin_probs)/len(nonlogin_probs)
  confidences = {"login": login_avg, "non-login":nonlogin_avg}
  processed_video = cv2.VideoCapture('output_video.mov')
  frames_counts = format_data(totalframes, lframes, dframes, nlframes)
  print(">>>>>>>>>>  ", processed_video)
  return processed_video, out_video_path, confidences, frames_counts