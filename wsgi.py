from app import app, warmup_embedding_models
from worker import start_all_workers

warmup_embedding_models()
start_all_workers()

if __name__ == "__main__":
    app.run()
