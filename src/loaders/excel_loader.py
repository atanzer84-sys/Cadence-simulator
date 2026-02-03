

def load_excel_parameters(excel_path, target_name):
    """Placeholder for Excel loading logic.

    In tests we monkeypatch this function, so the implementation can be
    filled in later without affecting the tests.
    """
    raise NotImplementedError("Excel loading not implemented yet")

    return star_params, planet_params, other_params

def _normalize_name(name):
    return name.lower().strip()
 