import numpy as np
import functions_framework
import os
from io import BytesIO
from gensim.models import Doc2Vec
from numpy.linalg import norm
from google.cloud import storage
from google.api_core.exceptions import NotFound
import fitz

storage_client = storage.Client()
model = Doc2Vec.load('cv_job_2.model')
bucket_name = os.getenv("BUCKET_NAME", "cv-bucket-dj02")
bucket = storage_client.bucket(bucket_name)

def read_pdf(filename):
    try:
        pdf = bucket.blob(filename).download_as_bytes()
        reader = fitz.open("pdf", BytesIO(pdf))
        return reader.load_page(0).get_text()
    except NotFound:
        return ""
    

def calculate_similarity(resume, jd_text):
    v1 = model.infer_vector(resume.split())
    v2 = model.infer_vector(jd_text.split())
    similarity = 100 * (np.dot(np.array(v1), np.array(v2))) / (norm(np.array(v1)) * norm(np.array(v2)))
    similarity = max(similarity, 0) 
    return round(similarity, 2)


@functions_framework.http
def send_scoring_result(request):
    request_json = request.get_json(silent=True)

    if "cv_name" not in request_json or "jobdesc_text" not in request_json:
        return {"message": "body should include cv_name and jobdesc_text fields"} 

    cv_name = request_json["cv_name"]
    jobdesc_text = request_json["jobdesc_text"]
    
    resume = read_pdf(cv_name)
    
    match_percentage = calculate_similarity(resume, jobdesc_text)

    return {"result": match_percentage, "resume": resume}