# Suppose we start with a list of bounding box positions:
# Each item = [x, y, width, height]
# Some values are missing (None / NaN)

positions = [
    [10, 20, 50, 60],
    [None, None, None, None],   # missing frame
    [30, 40, 55, 65],
    [None, 50, None, 70],       # partially missing
    [50, 60, 60, 80]
]

# Convert list into pandas DataFrame for easy processing
df = pd.DataFrame(positions, columns=['x', 'y', 'w', 'h'])

# DataFrame now looks like:
#     x     y     w     h
# 0  10.0  20.0  50.0  60.0
# 1   NaN   NaN   NaN   NaN
# 2  30.0  40.0  55.0  65.0
# 3   NaN  50.0   NaN  70.0
# 4  50.0  60.0  60.0  80.0


# Fill missing values using linear interpolation
df = df.interpolate(method='linear')

# After interpolation:
# Row 1 gets values between row 0 and row 2:
# x: 10 → 30 → becomes 20
# y: 20 → 40 → becomes 30
# w: 50 → 55 → becomes 52.5
# h: 60 → 65 → becomes 62.5
#
# Row 3:
# x: between 30 and 50 → 40
# y: already 50 (kept)
# w: between 55 and 60 → 57.5
# h: between 65 and 80 → 72.5


# Handle missing values at beginning and end
df = df.bfill().ffill()

# bfill = backward fill (fills from next valid value)
# ffill = forward fill (fills from previous valid value)
# This ensures if first or last rows were NaN, they get filled


# Replace infinite values with NaN (safety step)
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Example:
# If any value was accidentally inf or -inf, it becomes NaN


# Replace any remaining NaN with 0
df.fillna(0, inplace=True)

# Ensures no missing values remain


# Convert to integer numpy array (for drawing on image)
interpolated_positions = df.round().astype(int).to_numpy()

# Final result (rounded integers):
# [
#  [10, 20, 50, 60],
#  [20, 30, 52, 62],
#  [30, 40, 55, 65],
#  [40, 50, 58, 72],
#  [50, 60, 60, 80]
# ]

# This format is now ready for use in image processing
# (e.g., drawing bounding boxes frame by frame)