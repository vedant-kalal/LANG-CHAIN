import json
import os

nb_path = r"c:\Github Projects\LangChain\RAG_OVERVIEW.ipynb"

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cell = {
 "cell_type": "markdown",
 "metadata": {},
 "source": [
  "## RAG Architecture Diagram\n",
  "\n",
  "![RAG Architecture Diagram](images/rag_diagram.png)"
 ]
}

nb['cells'].append(new_cell)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Notebook updated successfully.")
