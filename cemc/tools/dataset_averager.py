import numpy as np
from scipy.interpolate import interp1d

class DatasetAverager(object):
    """
    Class that averages datasets where the range of the individual datasets
    may vary
    """
    def __init__(self, x_values):
        self.x_values = x_values
        self.y_values = np.zeros_like(self.x_values)
        self.y_values_squared = np.zeros_like(self.x_values)
        self.num_visits = np.zeros_like(self.x_values)

    def _get_boundary_indices(self, x_vals):
        """
        Finds which indices in the reference x-value array should be included
        """
        min_indx = None
        max_indx = None
        for i, value in enumerate(self.x_values):
            if min_indx is None and value >= np.min(x_vals):
                min_indx = i
            if max_indx is None and value >= np.max(x_vals):
                max_indx = i
        return min_indx, max_indx

    def add_dataset(self, x_vals, y_vals):
        """
        Add a new dataset to the array
        """
        interpolator = interp1d(x_vals, y_vals)
        min_indx, max_indx = self._get_boundary_indices(x_vals)
        if min_indx is None or max_indx is None:
            raise ValueError("X-values does not overlap!")

        x_interp = self.x_values[min_indx:max_indx]
        y_interp = interpolator(x_interp)
        self.y_values[min_indx:max_indx] += y_interp
        self.y_values_squared[min_indx:max_indx] += y_interp**2
        self.num_visits[min_indx:max_indx] += 1

    def _remove_unvisited(self):
        """
        Remove all the unvisited sizes
        """
        self.x_values = self.x_values[self.num_visits > 0]
        self.y_values = self.y_values[self.num_visits > 0]
        self.y_values_squared = self.y_values_squared[self.num_visits > 0]
        self.num_visits = self.num_visits[self.num_visits > 0]

    def get(self):
        """
        Get results
        """
        self._remove_unvisited()
        res = {}
        res["x_values"] = self.x_values
        res["y_values"] = self.y_values/self.num_visits
        res["std_y"] = np.sqrt(self.y_values_squared/self.num_visits -
                               res["y_values"]**2)
        res["num_visits"] = self.num_visits
        return res

    def _get_default_field_dict(self):
        """Returns a trivial field dictionary."""
        field_dict = {
            "x_values": "x_values",
            "y_values": "y_values",
            "std_y": "std_y",
            "num_visits": "num_visits"
        }
        return field_dict

    def _get_time_stamp(self):
        """Returns the timestamp."""
        import time
        import datetime
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        return st

    def save_to_db(self, db_name=None, table=None, name="noname", fields=None,
                   info={}):
        """Saves the results into a database."""
        import dataset
        if db_name is None:
            raise ValueError("No database name given!")
        if table is None:
            table = "dataset_averager"
        res = self.get()

        field_dict = self._get_default_field_dict()

        db = dataset.connect("sqlite:///{}".format(db_name))
        tbl = db[table]
        if fields is not None:
            for key, value in fields.items():
                field_dict[key] = value

        stamp = self._get_time_stamp()
        rows = []
        for i in range(len(self.x_values)):
            db_entry = {
                "name": name,
                "timestamp": stamp
            }
            db_entry[field_dict["x_values"]] = res["x_values"][i]
            db_entry[field_dict["y_values"]] = res["y_values"][i]
            db_entry[field_dict["std_y"]] = res["std_y"][i]
            db_entry[field_dict["num_visits"]] = res["num_visits"][i]
            db_entry.update(info)
            rows.append(db_entry)
        tbl.insert_many(rows)
