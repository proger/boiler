from pathlib import Path

import annoy


class Nearest:
    def __init__(self, index_file: Path, metadata_file: Path, index_dim: int, index_kind: str):
        super().__init__()

        self.index = annoy.AnnoyIndex(index_dim, index_kind)
        self.index.load(str(index_file))

        self.metadata = list(enumerate(metadata_file.read_text()))

    def search(self, embedding, k=10):
        result = zip(*self.index.get_nns_by_vector(embedding, k, include_distances=True))
        return [dict(distance=dist, coub_id=self.metadata[i]) for i, dist in result]



