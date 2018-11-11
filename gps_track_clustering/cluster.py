import time
import pandas as pd
from collections import Counter
from geojson import Feature, Point, FeatureCollection
from abc import ABC, abstractmethod


class Cluster(ABC):
    """ Abstract base class Cluster, is implemented by subclasses:
        - ClusterStartLocs()
        - ClusterTrackSimilarity()

    Args:
        summary (pd.DataFrame): Each row in the DataFrame represents one activity.
                                Mandatory columns are: ['id', 'startpos_lat', 'startpos_lon',
                                     'distance'  'duration'  'elevation_gain']
    """

    def __init__(self, summary):
        """
        Constructor
        """

        # CONFIG
        pass

        # DATA
        self.desc    = ''
        self.summary = summary
        self.labels  = None
        self.cluster = None
        super().__init__()

    @abstractmethod
    def _apply_cluster_algo(self):
        """
        Abstract method needs to be implemented by subclasses
        """
        pass

    def do_clustering(self):
        """
        Executes the clustering process and assigns:
            - colors
            - names
            - centers
        """

        # Apply the clustering algorithm implemented by subclasses
        labels, n_clusters, cluster_ids = self._apply_cluster_algo()
        self.summary.loc[:, 'cluster_id'] = labels
        self.labels = labels

        # get Cluster Names
        map_id2name = self._get_cluster_names(cluster_ids)

        # get Cluster colors
        cluster_colors = self._get_cluster_colors(cluster_ids)

        # get Cluster centers
        cluster_center = self._get_cluster_centers()

        # Create cluster dict
        cluster = {}
        for cluster_id in cluster_ids:
            cluster[cluster_id] = {'name': map_id2name[cluster_id],
                                   'center': cluster_center[cluster_id],
                                   'color': cluster_colors[cluster_id],
                                   'n_activities': Counter(labels)[cluster_id]}

        self._set_cluster_summary(cluster)
        self.cluster = cluster

        # Print stats
        print('{}: {} Clusters identified:'.format(self.desc, n_clusters))
        for id, val in cluster.items():
            print('    - {} - {} - {}'.format(val['name'], val['n_activities'], val['color']))

    def get_overview_table(self):
        """
        Creates a html table summarizing the clusters

        Returns:
            str: String representig the html code
        """

        table = pd.DataFrame.from_dict(self.cluster, orient='index')
        table = table[['name', 'n_activities']].sort_values('name')
        table = table.reset_index()
        table = table.rename(columns={'name': 'Cluster Name',
                                      'index': 'Cluster ID',
                                      'n_activities': '# Tracks'})

        return table.to_html(index=False, classes=["table-bordered", "table-striped", "table-hover"])

    def get_startloc_geojson(self):
        """
        Create geojson from start locs

        Returns:
            geojson.FeatureCollection(): FeatureCollection containing all clustered start locations
        """

        features = []
        for id, row in self.summary.iterrows():
            cluster_id = row.cluster_id
            thisFeature = Feature(geometry=Point((row.startpos_lon, row.startpos_lat)),
                                  properties={'startpos_name': self.cluster[cluster_id]['name'],
                                              'color': self.cluster[cluster_id]['color']})
            features.append(thisFeature)
        feature_collection = FeatureCollection(features)

        return feature_collection

    def get_cluster_centers_list(self):
        """
        Returns a string representation of all start locations
        """

        centers = []
        for id, val in self.cluster.items():
            if id == -1:
                continue
            centers.append(val['center'])

        return centers

    def _get_cluster_centers(self):
        """
        Returns a dict representation of all start locations
        """
        centers = {}
        for cluster_id, group in self.summary.groupby('cluster_id'):
            if cluster_id == -1:
                centers[cluster_id] = None
                continue
            centers[cluster_id] = [group.startpos_lat.mean(), group.startpos_lon.mean()]

        return centers

    def _get_cluster_colors(self, cluster_ids):
        """
        Assigns to each cluster a distiguishable color.
        Not clustered will be black
        """
        import random
        import colorsys

        def HSVToRGB(h, s, v):
            (r, g, b) = colorsys.hsv_to_rgb(h, s, v)
            return (int(255 * r), int(255 * g), int(255 * b))

        def getDistinctColors(n):
            huePartition = 1.0 / (n + 1)
            rgbs = (HSVToRGB(huePartition * value, 1.0, 1.0) for value in range(0, n))
            hexs = ['#%02x%02x%02x' % (x, y, z) for (x, y, z) in rgbs]
            return hexs

        rainbow = getDistinctColors(len(cluster_ids) + 1)
        random.shuffle(rainbow)

        cols = {}
        for cluster_id in cluster_ids:
            if cluster_id == -1:
                cols[cluster_id] = 'black'
                continue
            cols[cluster_id] = rainbow.pop()
        return cols

    def _get_cluster_names(self, start_ids):
        """
        Assigns to each cluster a name based on the closest city.
        """
        from geopy.point import Point
        from geopy.geocoders import ArcGIS

        data = self.summary
        map_id2name = {}
        for cluster_id in start_ids:
            if cluster_id == -1:
                map_id2name[cluster_id] = 'notClustered'
                continue
            # get average startpoint of this cluster
            lat_mean = data[data['cluster_id'] == cluster_id]['startpos_lat'].mean()
            lon_mean = data[data['cluster_id'] == cluster_id]['startpos_lon'].mean()
            lon_mean = data[data['cluster_id'] == cluster_id]['startpos_lon'].mean()
            p1 = Point(lat_mean, lon_mean)

            # Get city name of this location
            attempts = 0
            while attempts < 3:
                try:
                    geolocator = ArcGIS()
                    city = geolocator.reverse(p1).raw['City']
                    break
                except Exception:
                    attempts += 1
                    time.sleep(3)
                    print('Call to geopy.ArcGIS() failed, retry...')

            # Get mean distance
            dist_str = str(round(data[data['cluster_id'] == cluster_id]['distance'].mean() / 1000, 2))
            name = city + '_' + dist_str

            # In case one name is used more the once, add counter to name
            i = 2
            while name in map_id2name.values():
                name = name + '_' + str(i)
                i = i + 1
            map_id2name[cluster_id] = name

        return map_id2name

    def _get_cluster_names_simple(self, start_ids):
        """
        Assigns to each cluster a simple name
        """
        map_id2name = {}
        for cluster_id in start_ids:
            if cluster_id == -1:
                map_id2name[cluster_id] = 'notClustered'
                continue
            map_id2name[cluster_id] = 'Cluster_{}'.format(cluster_id)

        return map_id2name

    def _set_cluster_summary(self, cluster):
        """
        Set cluster information to self.summary object
        """
        map_id2name = {x: y['name'] for (x, y) in cluster.items()}
        map_id2color = {x: y['color'] for (x, y) in cluster.items()}
        self.summary['cluster_name'] = self.summary['cluster_id'].replace(map_id2name)
        self.summary['cluster_color'] = self.summary['cluster_id'].replace(map_id2color)
