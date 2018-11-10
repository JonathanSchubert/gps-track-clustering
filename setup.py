# flake8: noqa
from setuptools import setup
from distutils.util import convert_path

main_ns = {}
ver_path = convert_path('gps_track_clustering/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

setup(name              = 'gps_track_clustering',
      version           = main_ns['__version__'],
      description       = 'Clustering of GPS tracks',
      keywords          = 'GPS gpx track cluster clustering route routes',
      url               = None,
      author            = 'Jonathan Schubert',
      author_email      = None,
      license           = None,
      packages          = ['gps_track_clustering'],
      install_requires  = [
          'geojson',
          'sklearn',
          'geopy',
          'gpxpy',
      ],
      zip_safe          = False)
