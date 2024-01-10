from src.commander import AssertionCommander
from src.data_loader import AtlasDataLoader

batch_size = 3  # Or whatever chunk size you want
top_k = 5
url = "http://localhost:8080/"


commander = AssertionCommander(url, "double-transformers", batch_size, top_k)
print(commander.evaluate())
