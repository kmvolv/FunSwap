# FunSwap

---
## Introduction
---
Face-Swapping application for photos and real-time use. This project has been implemented using the [mediapipe](https://developers.google.com/mediapipe) library and the GUI of our application has been implemented with the help of the [Kivy Framework](https://kivy.org)

## How the application works
- The main working principle behind the solution is [Delaunay Triangulation](https://en.wikipedia.org/wiki/Delaunay_triangulation) which has been implemented with the help of the in-built functions present in Mediapipe and OpenCV. 
- Once the face outline is detected and segmented into triangles using the aforementioned triangulation method for both the source and the destination image, the triangles are then swapped with one another, after applying the appropriate amount of warping.
- For the final step, the swapped images are color corrected to match the tone of the original image, in order to make the outlines more visible. 

---
## Interface
---
### PicSwap
- Swaps the faces of two uploaded images.
- Demonstration :

### VidSwap
- Carries out real-time face swapping for a given reference image through users camera.
- Demonstration : 
