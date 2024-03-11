from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

model1 = "sentiment-analysis"
model2 = "bertweet-base-sentiment-analysis"
sentiment_pipeline = pipeline(model1)

app = FastAPI()

data = ["I love you", "I hate you"]
print(sentiment_pipeline(data))


class RequestModel(BaseModel):
    input_string: str

@app.post("/analyze")
def f(request: RequestModel):

    input_string = request.input_string

    sentiment = sentiment_pipeline(input_string)
    return {"result":
                {"sentiment": sentiment[0]["label"],
                 "score": sentiment[0]["score"]
                 }
            }