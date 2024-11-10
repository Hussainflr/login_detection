# import gradio as gr
# def greet(name, intensity):
#     return "Hello, " + name + "!" * int(intensity)

# demo = gr.Interface(
#     fn=greet,
#     inputs=["text", "slider"],
#     outputs=["text"],
# )

# demo.launch(share=True)


import requests
from PIL import Image
from torchvision import transforms
import torch
from ultralytics import YOLO
# model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True).eval()
# # Download human-readable labels for ImageNet.
# response = requests.get("https://git.io/JJkYN")
# labels = response.text.split("\n")

from aliyunsdkcore import client
from aliyunsdkgreen.request.v20180509 import ImageSyncScanRequest
from aliyunsdkgreenextension.request.extension import HttpContentHelper
import json
import uuid
from aliyunsdkcore import client
from aliyunsdkgreen.request.v20180509 import ImageSyncScanRequest
import json
import uuid
import oss2
from oss2 import Bucket, Auth
import re 
import os
from PIL import Image
import io

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

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
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_video_path = 'output_video.mov'
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
    processed_video = cv2.VideoCapture('output_video.mov')
    print(">>>>>>>>>>  ", processed_video)
    return processed_video, out_video_path


def format_data(totalframes, lframes, dframes, nlframes):
  # Format the dictionary as a colored string for Markdown
  formatted_data = "\n".join([
    f"<span style='color: blue; font-weight: bold'>Total frames</span>: <span style='color: black'>{totalframes}</span><br>",
    f"<span style='color: green; font-weight: bold'>Login Frames</span>: <span style='color: black'>{lframes}</span><br>",
    f"<span style='color: orange; font-weight: bold'>Duplicate Login Frames</span>: <span style='color: black'>{dframes}</span><br>",
    f"<span style='color: red; font-weight: bold'>Nonlogin Frames</span>: <span style='color: black'>{nlframes}</span>"
  ])
  return formatted_data


def extract_login_frames(input_video, similarity_threshold=0.7):
  model = YOLO('runs/classify/train17/weights/best.pt')
  input_video = cv2.VideoCapture(input_video)
  frame_rate = input_video.get(cv2.CAP_PROP_FPS)
  width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
  totalframes = input_video.get(cv2.CAP_PROP_FRAME_COUNT)

  # Create VideoWriter object to output the processed video
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out_video_path = 'output_video.mov'
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


def getemails(text:str):
  emails = re.findall(r'[\w\.-]+@[\w\.-]+(?:\.[\w]+)+', text)
  return emails



def uploadimg(img):
  new_file_name = f"{uuid.uuid4()}.png"
  # 阿里云主账号AccessKey拥有所有API的访问权限，风险很高。强烈建议您创建并使用RAM账号进行API访问或日常运维，请登录 https://ram.console.aliyun.com 创建RAM账号。
  access_key_id = ''
  access_key_secret = ''
  bucket_name = 'hussain'
  endpoint = 'oss-cn-beijing.aliyuncs.com' # 根据Bucket所在地域填写，例如'oss-cn-hangzhou.aliyuncs.com'
  if isinstance(img, np.ndarray):
    img = Image.fromarray(img)

  # # Save image to a byte buffer
  buffer = io.BytesIO()
  img.save(buffer, format='png')  # You can change format to PNG, etc.
  buffer.seek(0)  # Rewind the buffer to the beginning
  # 创建Bucket实例
  auth = oss2.Auth(access_key_id, access_key_secret)
  bucket = oss2.Bucket(auth, endpoint, bucket_name)
  # 上传图片
  object_key = 'temp/'+new_file_name  # 图片在Bucket中的存储路径和文件名
  bucket.put_object(object_key, buffer)
 

  # 获取图片URL
  if bucket.PUBLIC_ACCESS_BLOCK:
      # 如果Bucket是公共读，则可以直接构造URL
      url = f'https://{bucket_name}.{endpoint}/{object_key}'
  else:
      # 如果Bucket不是公共读，则需要生成一个签名的URL
      expires = 3600  # URL有效期，单位为秒
      url = bucket.sign_url('GET', object_key, expires)
  print(">>>>>>>>>>> Uploaded ", url)
  return url


def upload_video_frames(input_video):
  frames_url_list = []
  while True:
    ret, current_frame = input_video.read()
    if not ret:
        break  # End of video
    frame_url = uploadimg(current_frame)
    frames_url_list.append(frame_url)

  return frames_url_list
    
      

    
    

def ocr(im_url):
    clt = client.AcsClient("", "")
    #coding=utf-8
    # The following code provides an example on how to call the ImageSyncScanRequest operation to submit synchronous OCR tasks and obtain the moderation results in real time. 

    # Note: We recommend that you reuse the instantiated client as much as possible. This improves moderation performance and avoids repeated client connections. 
    # Common ways to obtain environment variables:
    # Obtain the AccessKey ID of your RAM user: os.environ['ALIBABA_CLOUD_ACCESS_KEY_ID']
    # Obtain the AccessKey secret of your RAM user: os.environ['ALIBABA_CLOUD_ACCESS_KEY_SECRET']

    request = ImageSyncScanRequest.ImageSyncScanRequest()
    request.set_accept_format('JSON')
    task = {"dataId": str(uuid.uuid1()),
        "url":im_url
        }

    print(task)
    # Create one task for each image to be moderated. 
    # If you moderate multiple images in a request, the total response time that the server spends processing the request begins when the request is initiated and ends upon moderation of the last image. 
    # In most cases, the average response time of moderating multiple images in a request is longer than that of moderating a single image. The more images you submit at a time, the higher the probability that the average response time is extended. 
    # In this example, a single image is moderated. If you want to moderate multiple images at a time, create one task for each image to be moderated. 
    # The OCR expense equals the product of the number of images moderated and the moderation unit price. 
    request.set_content(HttpContentHelper.toValue({"tasks": [task],
                        "scenes": ["ocr"]
                        }))
    response = clt.do_action_with_exception(request)
    print(response)
    result = json.loads(response)
    #TODO correct this logi
    if 200 == result["code"]:
        taskResults = result["data"]
        for taskResult in taskResults:
            if (200 == taskResult["code"]):
                sceneResults = taskResult["results"]
                print(sceneResults)
                return sceneResults[0]["ocrData"][0]
            else:
                return f'Error: {taskResult["code"]}'
    return f'Error: {result["code"]}'


def predict(video):
  # video, video_path = remove_similar_frames(video)
  processed_video, video_path, confidences, frames_count = extract_login_frames(video)
  img_urls = upload_video_frames(processed_video)
  # model = YOLO('runs/classify/train17/weights/best.pt')
  # results = model(img)
  # probs = results[0].probs.data.numpy()
  # login = probs[0]
  # nonlogin = probs[1]
  # confidences = {"login": login, "non-login":nonlogin}
  emails = ["No Emails Found"]
  text = ""
  for img_url in img_urls:
      try:
        text = ocr(img_url)
        emails = getemails(text=text)
      except:
         emails = ["OCR Failed Please Try Again."]
  if not emails:
     emails = ["No Emails Found"]
  # else:
  #   print("Email ", emails)
  #   confidences = {"login": probs[0], "non-login":probs[1]}
  
  return confidences, emails,frames_count, video_path

import gradio as gr

demo = gr.Interface(fn=predict,
             inputs=gr.Video(label="Input Video"),
             outputs=[gr.Label(num_top_classes=2, label="Confidence Scores"), gr.Textbox(label="Extracted Emails"),gr.Markdown(label="Frames stats"), gr.Video(label="Predicted Login frames")],
             examples=["test1.mov", "test2.mov"])

demo.launch(share=True)
