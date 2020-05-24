Components
==========

LASIF is organized into multiple components which handle various things.
The standard user should not need to interact with these components because
the `API <./api_doc.html>`_ and the `CLI <./cli.html>`_ should cover all
the needed functionalities to run a full-waveform inversion.

However, if anyone wants to do some more advanced scripting, one can
interact with the components of LASIF directly. In order to do so, one
would need to access the LASIF communicator object and with that, everything
should be possible. The LASIF communicator object is specific to each
project and can be accessed via the API interface.

The documentation for each component can be accessed from via the links
here below.

.. toctree::
    :maxdepth: 1
    
    components/project
    components/adjoint_sources
    components/downloads
    components/events
    components/iterations
    components/query
    components/validator
    components/visualizations
    components/waveforms
    components/weights
    components/windows
