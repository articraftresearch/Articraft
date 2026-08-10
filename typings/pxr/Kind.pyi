from typing import Any

class Tokens:
    def __getattr__(self, name: str) -> Any: ...
    assembly: str
    component: str
