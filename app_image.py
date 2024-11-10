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

  # image = Image.fromarray(img)

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


def predict(img):
  img_url = uploadimg(img)

  model = YOLO('runs/classify/train17/weights/best.pt')
  results = model(img)
  probs = results[0].probs.data.numpy()
  login = probs[0]
  nonlogin = probs[1]
  confidences = {"login": login, "non-login":nonlogin}
  emails = ["No Emails Found"]
  text = ""
  if login > 0.79:
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
  
  return confidences, emails

import gradio as gr

demo = gr.Interface(fn=predict,
             inputs=gr.Image(type="pil"),
             outputs=[gr.Label(num_top_classes=3), gr.Textbox()],
             examples=["test15.png", "test16.png"])

demo.launch()
