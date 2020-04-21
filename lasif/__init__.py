#!/usr/bin/env python
# -*- coding: utf-8 -*-
from lasif import api
from lasif.exceptions import (
    LASIFAdjointSourceCalculationError,
    LASIFCommandLineException,
    LASIFError,
    LASIFNotFoundError,
    LASIFWarning,
)
from .version import get_git_version

__all__ = [
    "api",
    "LASIFAdjointSourceCalculationError",
    "LASIFError",
    "LASIFNotFoundError",
    "LASIFWarning",
    "LASIFCommandLineException",
]

__version__ = get_git_version()
