from app import app
from worker import start_all_workers

start_all_workers()

if __name__ == "__main__":
    app.run()
