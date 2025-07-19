# Autonomous Shelf Robot

FYP projects made to tackle Out-Of-Stock (OOS) issues in retail stores. Utilizies YOLOv11 as the vision model with ROS2 and Gazebo for the robot simulation and communication.

## Software Implementation

### 1. Distange Regulation Node

This node is used to maintain a distance of 1.7m between the shelf and the robot. The ditance can be adjusted to fit the whole shelf within the frame of the camera.

Uses P-control to regulate the distance.

![Alt Text](Images/27.%20Distance%20Regulation%20Node.JPG)
![Alt Text](Images/31%20LiDAR%20Angle.JPG)

### 2. YOLO Node

This node is used to run inference from the images taken from the "image_raw" topicc.

![Alt Text](Images/34.%20Yolo%20Node.JPG)

![Alt Text](Images/33.%20Yolo.JPG)


### 3. Retail Vision System Implementation

#### Main Case
- When the whole product is missing. Whole refers to no product of the same class as the missing product is present in the row.

![Alt Text](Images/11.%20main%20case%20example.JPG)

Algorithm Used

![Alt Text](Images/22.%20main%20case%20algo%20example.JPG)

#### First Edge Case
- When the product is partially missing at the start of the row. Partially refers to having a product that is the same class as the missing product present in the row.

![Alt Text](Images/14.%20first%20edge%20case%20example.JPG)

Algorithm Used

![Alt Text](Images/23.%20first%20test%20case%20algo%20example.JPG)

#### Second Edge Case
- When the product is partially missing at the end of the row. Partially refers to having a product that is the same class as the missing product present in the row.

![Alt Text](Images/17.%20second%20edge%20case%20example.JPG)

Algorithm Used

![Alt Text](Images/24.%20second%20test%20case%20algo%20example.JPG)

#### Third Edge Case
- When the product is partially missing in the row except first and last position. Partially refers to having a product that is the same class as the missing product present in the row.

![Alt Text](Images/20.%20third%20edge%20case%20example.JPG)

Algorithm Used

![Alt Text](Images/25.%20third%20test%20case%20algo%20example.JPG)

### 4. Planogram Generation

When the amount of empty products is recognized, the planogram is then generated and compared with the provided planogram. This allows the recognition of what and how many of the products are missing.

# How to use
Clone the repo
```
git clone https://github.com/IlhanHashimEng/AutonomousShelfRobot.git
```

Move into src folder

```
cd src
```

Use MAKE to run the script

```
MAKE SOURCE=empty.mov
```