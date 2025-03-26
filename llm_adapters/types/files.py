from os import PathLike
from typing import IO, Mapping, Optional, Tuple, TypedDict, Union
import httpx
from llm_adapters.types.openai import Body, Headers, Query, NotGiven, FileObject

FileContent = Union[IO[bytes], bytes, PathLike[str]]

FileTypes = Union[
    # file (or bytes)
    FileContent,
    # (filename, file (or bytes))
    Tuple[Optional[str], FileContent],
    # (filename, file (or bytes), content_type)
    Tuple[Optional[str], FileContent, Optional[str]],
    # (filename, file (or bytes), content_type, headers)
    Tuple[Optional[str], FileContent, Optional[str], Mapping[str, str]],
]


class ExtraKwargs(TypedDict, total=False):
    extra_headers: Headers | None
    extra_query: Query | None
    extra_body: Body | None
    timeout: float | httpx.Timeout | None | NotGiven


__all__ = ["FileObject", "ExtraKwargs", "FileTypes"]
