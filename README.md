![Logo](/doc/images/logo/lasif_logo.png)
---
[![Build Status](https://api.travis-ci.org/dirkphilip/LASIF_2.0.png?branch=master)](https://travis-ci.org/dirkphilip/LASIF_2.0)
[![GPLv3](http://www.gnu.org/graphics/gplv3-88x31.png)](https://github.com/dirkphilip/LASIF_2.0/blob/master/LICENSE)


Detailed documentation: [LASIF](http://dirkphilip.github.io/LASIF_2.0)

Installation process:

```bash
cd <directory to download LASIF>
git clone https://github.com/dirkphilip/LASIF_2.0.git
cd LASIF_2.0
conda env create -f environment.yml
conda activate lasif
pip install -e .
```


## Paper

If you use LASIF for your project, please consider citing our paper(s):

>*Solvi Thrastarson, Dirk-Philip van Herwaarden, Lion Krischer and Andreas Fichtner* (2021),
**LASIF: LArge-scale Seismic Inversion Framework, an updated version**, EarthArXiv, [doi:10.31223/X5NC84](https://doi.org/10.31223/X5NC84).

Bibtex:
```
@article{Thrastarson_2021,
	doi = {10.31223/x5nc84},
	url = {https://doi.org/10.31223%2Fx5nc84},
	year = {2021},
	month = {jan},
	publisher = {California Digital Library ({CDL})},
	author = {Solvi Thrastarson and Dirk-Philip van Herwaarden and Lion Krischer and Andreas Fichtner},
	title = {{LASIF}: {LArge}-scale Seismic Inversion Framework, an updated version}
}
```

For more details and the reasoning behind LASIF, please also see the paper associated with the original version of LASIF:

>*Lion Krischer, Andreas Fichtner, Saule Zukauskaite, and Heiner Igel* (2015),
**Large‐Scale Seismic Inversion Framework**, Seismological Research Letters, [doi:10.1785/0220140248](http://dx.doi.org/10.1785/0220140248).


Bibtex:
```
@article{krischer2015large,
  title={Large-scale seismic inversion framework},
  author={Krischer, Lion and Fichtner, Andreas and Zukauskaite, Saule and Igel, Heiner},
  journal={Seismological Research Letters},
  volume={86},
  number={4},
  pages={1198--1207},
  year={2015},
  publisher={Seismological Society of America}
}
```
