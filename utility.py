from PIL import Image
import uuid
from oss2 import Bucket, Auth
import re 
from PIL import Image
import io
import numpy as np
from config import ALIBABA_CLOUD_ACCESS_KEY_ID, ALIBABA_CLOUD_ACCESS_KEY_SECRET,\
  ENDPOINT, BUCKET_NAME


def getemails(text:str):
  emails = re.findall(r'[\w\.-]+@[\w\.-]+(?:\.[\w]+)+', text)
  return emails

def format_data(totalframes, lframes, dframes, nlframes):
  # Format the dictionary as a colored string for Markdown
  formatted_data = "\n".join([
    f"<span style='color: blue; font-weight: bold'>Total frames</span>: <span style='color: black'>{totalframes}</span><br>",
    f"<span style='color: green; font-weight: bold'>Login Frames</span>: <span style='color: black'>{lframes}</span><br>",
    f"<span style='color: orange; font-weight: bold'>Duplicate Login Frames</span>: <span style='color: black'>{dframes}</span><br>",
    f"<span style='color: red; font-weight: bold'>Nonlogin Frames</span>: <span style='color: black'>{nlframes}</span>"
  ])
  return formatted_data


def uploadimg(img):
  new_file_name = f"{uuid.uuid4()}.png"
  if isinstance(img, np.ndarray):
    img = Image.fromarray(img)

  # # Save image to a byte buffer
  buffer = io.BytesIO()
  img.save(buffer, format='png')  # You can change format to PNG, etc.
  buffer.seek(0)  # Rewind the buffer to the beginning
  # 创建Bucket实例
  auth = Auth(ALIBABA_CLOUD_ACCESS_KEY_ID, ALIBABA_CLOUD_ACCESS_KEY_SECRET)
  bucket = Bucket(auth, ENDPOINT, BUCKET_NAME)
  # 上传图片
  object_key = 'temp/'+new_file_name  # 图片在Bucket中的存储路径和文件名
  bucket.put_object(object_key, buffer)
 

  # 获取图片URL
  if bucket.PUBLIC_ACCESS_BLOCK:
      # 如果Bucket是公共读，则可以直接构造URL
      url = f'https://{BUCKET_NAME}.{ENDPOINT}/{object_key}'
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