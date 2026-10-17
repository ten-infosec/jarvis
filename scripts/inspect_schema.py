import sys
import json


def describe(value, indent=0, depth=0):
    pad = "  " * indent
    if depth > 6:
        print(pad + "...")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            print(f"{pad}{key}: {type(child).__name__}")
            describe(child, indent + 1, depth + 1)
    elif isinstance(value, list):
        print(f"{pad}[list of {len(value)} items] -> first item:")
        if value:
            describe(value[0], indent + 1, depth + 1)


path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    data = json.load(f)
print(f"TOP: {type(data).__name__}")
describe(data)