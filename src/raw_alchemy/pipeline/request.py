class ProcessRequest:
    """Immutable processing request. Eliminates race conditions."""
    def __init__(self, path: str, params: dict, request_id: int):
        self.path = path
        self.params = params.copy()  # Defensive copy
        self.request_id = request_id
