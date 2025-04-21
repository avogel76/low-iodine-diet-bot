# build_embeddings.py
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path

MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
TEXT_PATH = 'knowledge/base.txt'
OUTPUT_PATH = 'embeddings/base.pkl'

def build():
    model = SentenceTransformer(MODEL_NAME)
    path = Path(TEXT_PATH)
    lines = [line.strip() for line in path.read_text(encoding='utf-8').split('\n') if line.strip()]
    embeddings = model.encode(lines, convert_to_tensor=True)
    
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump((embeddings, lines), f)

    print(f'✅ נוצרו {len(lines)} משפטים ונשמרו ל-{OUTPUT_PATH}')

if __name__ == '__main__':
    build()
