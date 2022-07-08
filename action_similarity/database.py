from bpe import Config

from action_similarity.utils import exist_embeddings, parse_action_label, load_embeddings


class ActionDatabase():

    def __init__(
        self,
        config: Config,
        database_path: str,
        label_path: str,
        target_actions=None):
        assert exist_embeddings(
            config=config, 
            embeddings_dir=database_path), \
                f"The embeddings(k = {config.k_clusters}) not exist. "\
                f"You should run the main with --update or bin.postprocess with --k_clusters option"
        
        self.db = load_embeddings(
            config=config,
            embeddings_dir=database_path,
            target_actions=target_actions,
        )
        self.actions = parse_action_label(label_path)