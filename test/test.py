import math
import numpy as np
import math3d as m3d
import re
from typing import List

def extract_coordinates_from_orientation(oriented_xyz: m3d.Transform) -> List[float]:
    oriented_xyz_coord = oriented_xyz.pose_vector
    coordinates_str = str(oriented_xyz_coord)
    numbers = re.findall(r"-?\d+\.\d+", coordinates_str)
    coordinates = [float(num) for num in numbers]
    return coordinates

position = [0,0,1,1,1,2]
orig = m3d.Transform(position)
print('orig:', orig)
# origin = set_lookorigin()
# # Create vector for position
xyz_coords = m3d.Vector(0, 0, 2)
# print('xyz_coords:', xyz_coords)
tcp_rotation_rpy = m3d.Vector(2, 1, 0)
# print('tcp_rotation_rpy:', tcp_rotation_rpy)
tcp_orient = m3d.Orientation.new_euler(tcp_rotation_rpy, encoding='xyz')
# print('tcp_orient:', tcp_orient)
position_vec_coords = m3d.Transform(tcp_orient, xyz_coords)
# position_vec_coords = m3d.Transform(tcp_rotation_rpy, xyz_coords)

print('position_vec_coords:', position_vec_coords)
# Transform based on origin
oriented_xyz = orig * position_vec_coords
# oriented_xyz += orig
print('oriented_xyz:', extract_coordinates_from_orientation(oriented_xyz))
# coordinates = extract_coordinates_from_orientation(oriented_xyz)
