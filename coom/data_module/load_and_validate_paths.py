def validate_weights(data, level="top"):
    """Recursively validate that weights sum to 1 at each level."""
    total_weight = 0.0
    for _, value in data.items():
        total_weight += value['weight']

    if not abs(total_weight - 1.0) < 1e-6:
        raise ValueError(f"Weight sum error at {level} level. Found {total_weight}, expected 1.0")

def flatten_paths(config_section):
    """
    Flatten nested subcategory structure into NeMo's expected list:
    ["weight1", "path1", "weight2", "path2", ...]
    """
    flat_list = []

    # Validate top-level subcategory weights
    validate_weights(config_section['subcategories'], level="subcategories")

    for sub_name, sub_info in config_section['subcategories'].items():
        sub_weight = sub_info['weight']

        # Validate internal path weights
        path_total = sum(p['weight'] for p in sub_info['paths'])
        if not abs(path_total - 1.0) < 1e-6:
            raise ValueError(f"Internal weights in subcategory '{sub_name}' must sum to 1. Found {path_total}")

        for path_info in sub_info['paths']:
            final_weight = sub_weight * path_info['weight']
            flat_list.append(str(final_weight))
            flat_list.append(path_info['path'])

    return flat_list

def load_data_paths(config):
    """Converts config to NeMo compatible format."""

    final_paths = {}
    for split in ['train', 'validation', 'test']:
        if split in config:
            final_paths[split] = flatten_paths(config[split])

    return final_paths
