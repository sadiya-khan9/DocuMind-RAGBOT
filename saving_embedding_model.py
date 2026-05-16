from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
model.save("./local_bge_model")
print("Model saved locally!")