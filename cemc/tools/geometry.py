
class Plane(object):
    def __init__(self, normal, point_in_plane):
        self.normal = normal
        self.point_in_plane = point_in_plane

    def signed_distance(self, x):
        """Return the distance from x to the plane."""
        diff = x - self.point_in_plane
        return self.normal.dot(diff)

    def distance(self, x):
        """Return the distance from x to the plane."""
        return abs(self.signed_distance(x))