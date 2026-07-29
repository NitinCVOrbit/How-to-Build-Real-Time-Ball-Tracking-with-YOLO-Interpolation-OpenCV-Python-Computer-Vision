# ============================================================
# BALL TRACKING PIPELINE (COMMENT VERSION)
# ============================================================

# 1. Load the video using OpenCV
#    - Initialize VideoCapture
#    - Extract total number of frames

# 2. Run YOLO model on each frame
#    - For every frame:
#        • Detect objects using YOLO
#        • If ball is detected → store (x, y, w, h)
#        • If not detected → store NaN values
#    - Save results in a list (or load from cache if available)

# 3. Perform interpolation on missing detections
# 4. Visualize the tracking results
# 5. Release resources



# ============================================================
# IMPORTS
# ============================================================

import cv2                    
import numpy as np            
import pandas as pd           
from ultralytics import YOLO  
from tqdm import tqdm         
import os      
import pickle                 

# ============================================================
# STEP 1: USER SETTINGS
# ============================================================

VIDEO_PATH = "ball.mp4"       # Path to input video
MODEL_PATH = "best.pt"        # Path to trained YOLO model
BALL_CLASS_ID = 0             # Class ID representing the ball

CACHE_FILE = "ball_positions.pkl"   # File where detections will be stored


cap = cv2.VideoCapture(VIDEO_PATH)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_width = 1280
frame_height = 720

# ============================================================
# STEP 2: LOAD YOLO MODEL
# ============================================================

model = YOLO(MODEL_PATH)      # Load trained YOLO model into memory

# # ============================================================
# # STEP 3: CHECK CACHE (OPTIMIZATION STEP)
# # ============================================================

# # Check if detection results are already saved
if os.path.exists(CACHE_FILE):

    print("⚡ Loading cached detections...")  # Inform user

    # Open the pickle file in read-binary mode
    with open(CACHE_FILE, "rb") as f:

        # Load stored positions (list of [x, y, w, h])
        positions = pickle.load(f)

    # Get total frames from stored data
    frame_count = len(positions)

    print("✅ Loaded from cache")

else:
    # If cache does not exist → run YOLO detection
    print("🔍 Running YOLO detection (first time only)...")

    # Open video file
    cap = cv2.VideoCapture(VIDEO_PATH)

    # Get total number of frames in video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize list with NaN values (no detection initially)
    positions = [[np.nan, np.nan, np.nan, np.nan] for _ in range(frame_count)]

    # Loop through all frames with progress bar
    for i in tqdm(range(frame_count), desc="YOLO Detection", unit="frame"):

        # Read frame from video
        ret, frame = cap.read()

        # If frame not read properly → stop loop
        if not ret:
            break

        frame = cv2.resize(frame,(1280,720))

        # Run YOLO model on current frame
        results = model.predict(frame, verbose=False)

        # Check if any objects are detected
        if results[0].boxes is not None:

            # Loop through all detected objects
            for j, cls_id in enumerate(results[0].boxes.cls):

                # Check if object is BALL
                if int(cls_id) == BALL_CLASS_ID:

                    # Extract bounding box (x_center, y_center, width, height)
                    x, y, w, h = map(float, results[0].boxes.xywh[j][:4])

                    # Store detection in positions list
                    positions[i] = [x, y, w, h]

                    # Stop after first ball detection
                    break

    # Release video after detection
    cap.release()

    # Save detection results into pickle file
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(positions, f)

    print("💾 Detections saved to cache")

# ============================================================
# STEP 4: INTERPOLATION
# ============================================================

print("📈 Performing interpolation...")

# Convert list into pandas DataFrame for easy processing
df = pd.DataFrame(positions, columns=['x', 'y', 'w', 'h'])

# Fill missing values using linear interpolation
df = df.interpolate(method='linear')

# Handle missing values at beginning and end
df = df.bfill().ffill()

# Replace infinite values with NaN (safety step)
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Replace any remaining NaN with 0
df.fillna(0, inplace=True)

# Convert to integer numpy array (for drawing on image)
interpolated_positions = df.round().astype(int).to_numpy()

print("✅ Interpolation complete")

# ============================================================
# STEP 5: RENDER VIDEO
# ============================================================

# Re-open video for visualization
cap = cv2.VideoCapture(VIDEO_PATH)

frame_idx = 0   # Track current frame index

# Get original FPS of video
fps = cap.get(cv2.CAP_PROP_FPS)

# Convert FPS into delay (ms per frame)
delay = int(1000 / fps)


# Loop through video frames again
while True:

    # Read frame
    ret, frame = cap.read()

    # Stop if video ends
    if not ret:
        break
    
    frame = cv2.resize(frame,(1280,720))

    # Get interpolated position for current frame
    x, y, w, h = interpolated_positions[frame_idx]

    # Decide color:
    # GREEN = real detection
    # RED   = interpolated value
    if not np.isnan(positions[frame_idx][0]):
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)

    # Define triangle points above the ball
    pts = np.array([
        [x, y - 10],      # Bottom point
        [x + 8, y - 25],  # Top right
        [x - 8, y - 25]   # Top left
    ], np.int32)

    # Fill triangle with color
    cv2.fillPoly(frame, [pts], color)

    # Draw border around triangle
    cv2.polylines(frame, [pts], True, (0, 0, 0), 1)

    # Move to next frame index
    frame_idx += 1

    # Show output frame
    cv2.imshow("Ball Tracking", frame)

    # Wait according to FPS and check for 'q' key
    key = cv2.waitKey(delay) & 0xFF
    if key == ord('q'):
        break

print("📽️ Video End")

# ============================================================
# STEP 6: CLEANUP
# ============================================================

cap.release()              # Release video resource
cv2.destroyAllWindows()    # Close all OpenCV windows



