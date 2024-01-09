import sys
from typing import List

import requests

from commander import AssertionCommander
from src.metrics import (
    MetricComputer,
    CombinedMetricComputer,
)

from itertools import islice


batch_size = 10  # Or whatever chunk size you want
top_k = 5
url = "http://localhost:8080/"


commander = AssertionCommander(url,"double-transformers",batch_size,top_k)
print(commander.evaluate())
