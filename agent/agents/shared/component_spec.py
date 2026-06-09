from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class NodeComponent:
    """Shared component contract for graph node assembly."""

    name: str
    factory: Callable[..., Callable[..., dict[str, Any]]]

    def build(self, llm: Any):
        return self.factory(llm=llm)

