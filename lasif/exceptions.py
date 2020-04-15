
class LASIFError(Exception):
    """
    Base exception class for LASIF.
    """

    pass


class LASIFNotFoundError(LASIFError):
    """
    Raised whenever something is not found inside the project.
    """

    pass


class LASIFAdjointSourceCalculationError(LASIFError):
    """
    Raised when something goes wrong when calculating an adjoint source.
    """

    pass


class LASIFWarning(UserWarning):
    """
    Base warning class for LASIF.
    """

    pass


class LASIFCommandLineException(LASIFError):
    """
    Command line exception.
    """
    
    pass