import cv2

# Input and output file paths
input_path = 'IMG_7262.MOV'     # Change to your input video
output_path = 'test4.mp4'   # Output in .mp4 format

# Open the input video
cap = cv2.VideoCapture(input_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' is a common codec for MP4
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Read and write each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)

# Release everything
cap.release()
out.release()
print("Video conversion complete:", output_path)
