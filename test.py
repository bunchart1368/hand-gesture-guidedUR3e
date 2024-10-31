import math

# List of angles in degrees
degrees_list = [0, -90, -90, 60, 90, 180]

# Convert each degree to radians
radians_list = [round(math.radians(degree), 3) for degree in degrees_list]

print(str(radians_list))