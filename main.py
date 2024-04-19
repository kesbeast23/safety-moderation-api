from fastapi import FastAPI,File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import vision
from pydantic import BaseModel
import os,io
from typing import Optional, Sequence
from google.cloud import videointelligence_v1 as vi
from datetime import timedelta
from collections import Counter
from google.cloud import storage
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="key.json"
bucket_name="app-videos"
description = """
Detection API Docs 🚀

## Image

You are able to upload an image and detect the following:

* **Adult Content** 
* **Violence** 
* **Racy Content** 
* **Spoofed Content** 

## Video

You are able to upload a video and detect the following:

* **General Explicit Content** 


"""
app = FastAPI(
     title="Detection API",
    description=description,
)



app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class Image(BaseModel):
    url: str


@app.get("/")
async def root():
    return {"message": "Detection API"}

@app.post("/image/")
def check_image( my_file: UploadFile = File(...)):
    client = vision.ImageAnnotatorClient()

    content = my_file.file.read()
    image = vision.Image(content=content)

    response = client.safe_search_detection(image=image)
    safe = response.safe_search_annotation
    # Names of likelihood from google.cloud.vision.enums
    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE','LIKELY', 'VERY_LIKELY')
    if response.error.message:
        raise Exception(
            'error:{}'.format(
                response.error.message))
    
    return {'adult':likelihood_name[safe.adult],'medical': likelihood_name[safe.medical],'spoofed':likelihood_name[safe.spoof],'violence':likelihood_name[safe.violence],'racy':likelihood_name[safe.racy]}

@app.post("/video/")
def check_video( my_file: UploadFile = File(...)):
    try:
        content = my_file.file.read()
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        print(my_file.filename)
        blob = bucket.blob(my_file.filename) 
        blob.upload_from_string(content,content_type=my_file.content_type)
        blob.make_public()
        url = f'gs://app-videos/{my_file.filename}'
        print(url)

        video_client = vi.VideoIntelligenceServiceClient()
        features = [vi.Feature.EXPLICIT_CONTENT_DETECTION]
        segment = vi.VideoSegment(
        start_time_offset=timedelta(seconds=0),
        end_time_offset=timedelta(seconds=10),
        )


        request = vi.AnnotateVideoRequest(
            input_uri=url,
            features=features,
        )
        print(f'Processing video "{url}"...')
        operation = video_client.annotate_video(request)
        results = operation.result().annotation_results[0] 
        frames = results.explicit_annotation.frames
        likelihood_counts = Counter([f.pornography_likelihood for f in frames])
        print(f" Explicit content frames: {len(frames)} ".center(40, "-"))
        chances = []
        for likelihood in vi.Likelihood:
            print(f"{likelihood.name:<22}: {likelihood_counts[likelihood]:>3}")
            chances.append({f"{likelihood.name}":likelihood_counts[likelihood]})

        return {'Detection frames':len(frames),'chances':chances}
    except Exception as e:
        print(e)
        return {'error': str(e) }
