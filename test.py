import math
import numpy as np
import math3d as m3d

position = [0,0,1,1,1,2]
orig = m3d.Transform(position)
print('orig:', orig)
# origin = set_lookorigin()
# # Create vector for position
xyz_coords = m3d.Vector(0, 0, 2)
print('xyz_coords:', xyz_coords)
tcp_rotation_rpy = m3d.Vector(2, 1, 0)
print('tcp_rotation_rpy:', tcp_rotation_rpy)
tcp_orient = m3d.Orientation.new_euler(tcp_rotation_rpy, encoding='xyz')
print('tcp_orient:', tcp_orient)
position_vec_coords = m3d.Transform(tcp_orient, xyz_coords)
# position_vec_coords = m3d.Transform(tcp_rotation_rpy, xyz_coords)

print('position_vec_coords:', position_vec_coords)
# # Transform based on origin
oriented_xyz = orig * position_vec_coords
print('oriented_xyz:', oriented_xyz)
# coordinates = extract_coordinates_from_orientation(oriented_xyz)
