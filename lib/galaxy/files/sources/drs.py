import logging
import re
from typing import (
    cast,
    Dict,
    Optional,
)

from typing_extensions import Unpack

from galaxy.util.drs import fetch_drs_to_file
from . import (
    BaseFilesSource,
    FilesSourceOptions,
    FilesSourceProperties,
    PluginKind,
)

log = logging.getLogger(__name__)


class DRSFilesSourceProperties(FilesSourceProperties, total=False):
    url_regex: str
    force_http: bool
    http_headers: Dict[str, str]


class DRSFilesSource(BaseFilesSource):
    plugin_type = "drs"
    plugin_kind = PluginKind.drs

    def __init__(self, **kwd: Unpack[FilesSourceProperties]):
        kwds: FilesSourceProperties = dict(
            id="_drs",
            label="DRS file",
            doc="DRS file handler",
            writable=False,
        )
        kwds.update(kwd)
        props: DRSFilesSourceProperties = cast(DRSFilesSourceProperties, self._parse_common_config_opts(kwds))
        self._url_regex_str = props.pop("url_regex", r"^drs://")
        assert self._url_regex_str
        self._url_regex = re.compile(self._url_regex_str)
        self._force_http = props.pop("force_http", False)
        self._props = props

    def _realize_to(self, source_path, native_path, user_context=None, opts: Optional[FilesSourceOptions] = None):
        props = self._serialization_props(user_context)
        headers = props.pop("http_headers", {}) or {}
        fetch_drs_to_file(source_path, native_path, user_context, headers=headers, force_http=self._force_http)

    def _write_from(self, target_path, native_path, user_context=None, opts: Optional[FilesSourceOptions] = None):
        raise NotImplementedError()

    def score_url_match(self, url: str):
        match = self._url_regex.match(url)
        if match:
            return match.span()[1]
        else:
            return 0

    def _serialization_props(self, user_context=None) -> DRSFilesSourceProperties:
        effective_props = {}
        for key, val in self._props.items():
            effective_props[key] = self._evaluate_prop(val, user_context=user_context)
        effective_props["url_regex"] = self._url_regex_str
        effective_props["force_http"] = self._force_http
        return cast(DRSFilesSourceProperties, effective_props)


__all__ = ("DRSFilesSource",)
