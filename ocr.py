from aliyunsdkcore import client
from aliyunsdkgreen.request.v20180509 import ImageSyncScanRequest
from aliyunsdkgreenextension.request.extension import HttpContentHelper
import json
import uuid
from aliyunsdkcore import client
from aliyunsdkgreen.request.v20180509 import ImageSyncScanRequest

from config import ALIBABA_CLOUD_ACCESS_KEY_ID, ALIBABA_CLOUD_ACCESS_KEY_SECRET


def ocr(im_url):
    clt = client.AcsClient(ALIBABA_CLOUD_ACCESS_KEY_ID, ALIBABA_CLOUD_ACCESS_KEY_SECRET)
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