import os
from pathlib import Path
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

import boiler.api.nearest

model_dir = Path(os.environ['BOILER_MODEL_DIR'])

nearest = boiler.api.nearest.Nearest(model_dir)

app = FastAPI()

class ItemEmbedding(BaseModel):
    embedding: List[float]

    class Config:
        schema_extra = {
            "example":{
                "embedding": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.2601329982280731,0,0.34684398770332336,0,0,0,0,0,0,0,0,0.08671099692583084,0,0,0.08671099692583084,0,0,0,0,0.08671099692583084,0,0,0.867110013961792,0,0,0,0,0,0.08671099692583084,0,0,0,0,0,0.17342199385166168,0,0,0,0,0,0,0,0,0,0,0,0,0]
            }
        }


@app.post("/nearest")
async def nearest_embedding(item: ItemEmbedding, k: int = 12):
    return nearest.search(item.embedding, k=k)