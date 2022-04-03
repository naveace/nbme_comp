import sys
from project.data.data_loaders import get_clean_train_data
from tqdm import tqdm
from project.embedding.embedding_mechanism import CorpusEmbedder
if __name__ == "__main__":
    embedder = CorpusEmbedder(sys.argv[1])
    train = get_clean_train_data()
    train = train.drop_duplicates(subset=['pn_num'])
    for idx, r in tqdm(train.iterrows(), total=len(train), desc=f"Embedding with {sys.argv[1]}"):
        embedder.embed(r)
