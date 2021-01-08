Welcome to LASIF 2.0
====================

.. |br| raw:: html

    <br />

.. image:: images/logo/lasif_logo.*
    :width: 80%
    :align: center


LASIF (**LA**\ rge-scale **S**\ eismic **I**\ nversion **F**\ ramework 2.0) is a
data-driven end-to-end workflow tool to perform adjoint full seismic waveform
inversions.

**Github:** The code is developed at and available from its
`Github repository <http://github.com/dirkphilip/LASIF_2.0>`_.

For the initial version of LASIF, we refer to its 
`repository <http://github.com/krischer/LASIF>`_ and its corresponding 
`documentation <http://krischer.github.io/LASIF>`_.

If you use LASIF in your project, please consider citing our paper(s).
If you use this version of LASIF, please refer to this paper:

.. admonition:: Latest paper

    *Solvi Thrastarson, Dirk-Philip van Herwaarden, Lion Krischer and Andreas Fichtner* (2021) |br|
    **LASIF: LArge-scale Seismic Inversion Framework, an updated version**, |br|
    EarthArXiv. |br|
    `doi:10.31223/X5NC84 <https://doi.org/10.31223/X5NC84>`_

.. admonition:: bibtex

    @article{thrastarson2021lasif, |br|
        title={LASIF: LArge-scale Seismic Inversion Framework, an updated version}, |br|
        author={Thrastarson, Solvi and van Herwaarden, Dirk-Philip and Krischer, Lion and Fichtner, Andreas}, |br|
        journal={EarthArXiv}, |br|
        pages={1965}, |br|
        year={2021}, |br|
        publisher={Center for Open Science} |br|
        }


For more details, we refer to the paper associated with the original version
of LASIF.

.. admonition:: Original LASIF paper

    *Lion Krischer, Andreas Fichtner, Saule Zukauskaite, and Heiner Igel* (2015), |br|
    **Large‐Scale Seismic Inversion Framework**, |br|
    Seismological Research Letters, 86(4), 1198–1207. |br|
    `doi:10.1785/0220140248 <http://dx.doi.org/10.1785/0220140248>`_

.. admonition:: bibtex

    @article{krischer2015large, |br|
        title={Large-scale seismic inversion framework}, |br|
        author={Krischer, Lion and Fichtner, Andreas and Zukauskaite, Saule and Igel, Heiner}, |br|
        journal={Seismological Research Letters}, |br|
        volume={86}, |br|
        number={4}, |br|
        pages={1198--1207}, |br|
        year={2015}, |br|
        publisher={Seismological Society of America} |br|
        }


---------


Dealing with the large amounts of data present in modern full seismic waveform
inversions in an organized, reproducible and shareable way continues to be a
major difficulty potentially even hindering actual research. LASIF improves the
speed, reliability, and ease with which such inversion can be carried out.

Full seismic waveform inversion using adjoint methods evolved into a well
established tool in recent years that has seen many applications. While the
procedures employed are (to a certain extent) well understood, large scale
applications to real-world problems are often hindered by practical issues.

The inversions use an iterative approach and thus by their very nature
encompass many repetitive, arduous, and error-prone tasks. Amongst these are
data acquisition and management, quality checks, preprocessing, selecting time
windows suitable for misfit calculations, the derivation of adjoint sources,
model updates, and interfacing with numerical wave propagation codes.

The LASIF workflow framework is designed to tackle these problems. One major
focus of the package is to handle vast amount of data in an organized way while
also efficiently utilizing modern HPC systems. The use of a unified framework
enables reproducibility and an efficient collaboration on and exchange of
tomographic images.


.. toctree::
    :hidden:
    :maxdepth: 2

    installation 

    introduction
    tutorial

    api_doc
    cli
    components
    faq

