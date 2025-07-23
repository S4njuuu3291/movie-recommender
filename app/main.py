from fastapi import FastAPI
import pandas as pd

tags_embedding_df = pd.read_csv("../data/processed/tags_embedding.csv")

# app = FastAPI()