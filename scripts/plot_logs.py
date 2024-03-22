import numpy as np
import sys
import os
import matplotlib.pyplot as plt


if (len(sys.argv) != 6):
    print("Error: usage   python {} <avg-offset-file> <std-dev-file> <xy-offset-file> <radial-offset-file> <output-dir>".format(sys.argv[0]))
    sys.exit()

avg_offset_file = sys.argv[1]
std_dev_file = sys.argv[2]
xy_offset_file = sys.argv[3]
radial_offset_file = sys.argv[4]
output_dir = sys.argv[5]

# plot frames vs. y-offset
with open(avg_offset_file, 'r') as f:
    lines = f.readlines()

    offsets = []
    for line in lines:
        offsets.append(float(line.strip()))

frames = range(len(lines))

plt.plot(frames, offsets)
plt.title("Average Y-Offset")
plt.xlabel("frame")
plt.ylabel("y-offset")

output_file = os.path.join(output_dir, "avg_offset.png")
plt.savefig(output_file, dpi=300)
plt.close()


# plot frames vs. STD
with open(std_dev_file, 'r') as f:
    lines = f.readlines()

    stdev = []
    for line in lines:
        stdev.append(float(line.strip()))

frames = range(len(lines))

plt.plot(frames, stdev)
plt.title("Average Standard Deviation")
plt.xlabel("frame")
plt.ylabel("std")

output_file = os.path.join(output_dir, "std_dev.png")
plt.savefig(output_file, dpi=300)
plt.close()


# plot x_offset vs. y-offset
with open(xy_offset_file, 'r') as f:
    lines = f.readlines()

    x_offsets = []
    y_offsets = []
    for line in lines:
        l = line.strip().split(',')
        x_offsets.append(float(l[0]))
        y_offsets.append(float(l[1]))

plt.scatter(x_offsets, y_offsets, s=0.5)
plt.title("X-Offset vs. Y-Offset")
plt.xlabel("x-offset")
plt.ylabel("y-offset")

output_file = os.path.join(output_dir, "xy_offset.png")
plt.savefig(output_file, dpi=300)
plt.close()


# plot radius vs. y-offset
with open(radial_offset_file, 'r') as f:
    lines = f.readlines()

    r_offsets = []
    y_offsets = []
    for line in lines:
        l = line.strip().split(',')
        r_offsets.append(float(l[0]))
        y_offsets.append(float(l[1]))

plt.scatter(r_offsets, y_offsets, s=0.5)
plt.title("Radial Distance vs. Y-Offset")
plt.xlabel("l2-norm from center")
plt.ylabel("y-offset")

output_file = os.path.join(output_dir, "radial_offset.png")
plt.savefig(output_file, dpi=300)
plt.close()
