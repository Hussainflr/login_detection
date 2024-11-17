from ocr import ocr
from utility import getemails, upload_video_frames
from image_processing import extract_login_frames
import gradio as gr


def predict(video, modelname):
  processed_video, video_path, confidences, frames_count = extract_login_frames(video, modelname)
  img_urls = upload_video_frames(processed_video)
  emails = []
  text = ""
  for img_url in img_urls:
      try:
        text = ocr(img_url)
        emails += list(set(getemails(text=text)))
      except:
         emails += ["OCR Failed Please Try Again."]
  if not emails:
     emails = ["No Emails Found"]
  emails = list(set(emails))
  return confidences, emails,frames_count, video_path



demo = gr.Interface(fn=predict,
             inputs=[
                gr.Video(label="Input Video"), 
                gr.Dropdown(choices=["MODEL1", "MODEL2"], value="MODEL1", label="Select a Model")
                     ],
             outputs=[
                gr.Label(num_top_classes=2, label="Confidence Scores"),
                gr.Textbox(label="Extracted Emails"),
                gr.Markdown(label="Frames stats"),
                gr.Video(label="Predicted Login frames")
                ],
              examples=[["test1.mov"], ["test2.mov"]]
             
)

demo.launch()