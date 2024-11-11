import re
from loaders.dataloader import *
from loaders.vectorstore import *
from dotenv import load_dotenv

loader = load_loader("data/dataloaders/KB_dog_loader.pkl")
print(loader.docs[0].metadata)

special_terms = []
pattern = r"특약 이름\s?\[([^\]]+)\]"

for doc in loader.docs:
    matches = re.findall(pattern, doc.page_content)
    for match in matches:
        special_terms.append({"name": match})

print(special_terms)
