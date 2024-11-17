from utility import uploadimg, getemails
from model import getmodel
from ocr import ocr

def predict(img, modelname):
  img_url = uploadimg(img)

  model = getmodel(modelname)
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
             inputs=[gr.Image(type="pil"), gr.Dropdown(choices=["MODEL1", "MODEL2"], label="Select Model")],
             outputs=[gr.Label(num_top_classes=3), gr.Textbox()],
             examples=[["test15.png"], ["test16.png"]])

demo.launch()
