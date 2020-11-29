from pathlib import Path

import annoy


class Nearest:
    def __init__(self, model_dir: Path, index_dim: int = 64, index_kind: str = 'angular'):
        super().__init__()

        index_file = model_dir / 'annoy'
        metadata_file = model_dir / '00000/default/metadata.tsv'

        self.index = annoy.AnnoyIndex(index_dim, index_kind)
        self.index.load(str(index_file))

        self.metadata = list(enumerate(metadata_file.read_text()))

    def search(self, embedding, k=10):
        result = zip(*self.index.get_nns_by_vector(embedding, k, include_distances=True))
        return [dict(distance=dist, coub_id=self.metadata[i]) for i, dist in result]
