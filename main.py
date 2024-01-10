from src.commander import AssertionCommander
from src.data_loader import AtlasDataLoader

batch_size = 25  # Or whatever chunk size you want
top_k = 5
url = "http://localhost:8080/"
assertion_number = 10
model_name = "double-transformers"

commander = AssertionCommander(url, model_name, batch_size, top_k, assertion_number)
print(commander.evaluate())
