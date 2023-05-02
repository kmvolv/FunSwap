from kivymd.app import App

from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.popup import Popup

from kivy.properties import ObjectProperty
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.uix.filechooser import FileChooserListView
from kivy.clock import Clock
from kivy.graphics.texture import Texture

from plyer import filechooser

import cv2
import numpy as np
import mediapipe as mp
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

class FaceSwap():
    global src_image, src_image_gray, src_mask, src_landmark_points, src_np_points, src_convexHull, indexes_triangles

    def get_landmark_points(self,src_image):
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                return None
            if len(results.multi_face_landmarks) > 1:
                sys.exit("There are too much face landmarks")

            src_face_landmark = results.multi_face_landmarks[0].landmark
            landmark_points = []
            for i in range(468):
                y = int(src_face_landmark[i].y * src_image.shape[0])
                x = int(src_face_landmark[i].x * src_image.shape[1])
                landmark_points.append((x, y))

            return landmark_points

    def extract_index_nparray(self,nparray):
        index = None
        for num in nparray[0]:
            index = num
            break
        return index

    def get_triangles(self,convexhull, landmarks_points, np_points):
        rect = cv2.boundingRect(convexhull)
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(landmarks_points)
        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles, dtype=np.int32)

        indexes_triangles = []
        for t in triangles:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])

            index_pt1 = np.where((np_points == pt1).all(axis=1))
            index_pt1 = self.extract_index_nparray(index_pt1)
            index_pt2 = np.where((np_points == pt2).all(axis=1))
            index_pt2 = self.extract_index_nparray(index_pt2)
            index_pt3 = np.where((np_points == pt3).all(axis=1))
            index_pt3 = self.extract_index_nparray(index_pt3)

            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                triangle = [index_pt1, index_pt2, index_pt3]
                indexes_triangles.append(triangle)

        return indexes_triangles

    def triangulation(self,triangle_index, landmark_points, img=None):
        tr1_pt1 = landmark_points[triangle_index[0]]
        tr1_pt2 = landmark_points[triangle_index[1]]
        tr1_pt3 = landmark_points[triangle_index[2]]
        triangle = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

        rect = cv2.boundingRect(triangle)
        (x, y, w, h) = rect

        cropped_triangle = None
        if img is not None:
            cropped_triangle = img[y: y + h, x: x + w]

        cropped_triangle_mask = np.zeros((h, w), np.uint8)

        points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                        [tr1_pt2[0] - x, tr1_pt2[1] - y],
                        [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

        cv2.fillConvexPoly(cropped_triangle_mask, points, 255)

        return points, cropped_triangle, cropped_triangle_mask, rect

    def warp_triangle(self,rect, points1, points2, src_cropped_triangle, dest_cropped_triangle_mask):
        (x, y, w, h) = rect
        matrix = cv2.getAffineTransform(np.float32(points1), np.float32(points2))
        warped_triangle = cv2.warpAffine(src_cropped_triangle, matrix, (w, h))
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=dest_cropped_triangle_mask)
        return warped_triangle

    def add_piece_of_new_face(self,new_face, rect, warped_triangle):
        (x, y, w, h) = rect
        new_face_rect_area = new_face[y: y + h, x: x + w]
        new_face_rect_area_gray = cv2.cvtColor(new_face_rect_area, cv2.COLOR_BGR2GRAY)
        _, mask_triangles_designed = cv2.threshold(new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

        new_face_rect_area = cv2.add(new_face_rect_area, warped_triangle)
        new_face[y: y + h, x: x + w] = new_face_rect_area

    def swap_new_face(self,dest_image, dest_image_gray, dest_convexHull, new_face):
        face_mask = np.zeros_like(dest_image_gray)
        head_mask = cv2.fillConvexPoly(face_mask, dest_convexHull, 255)
        face_mask = cv2.bitwise_not(head_mask)

        head_without_face = cv2.bitwise_and(dest_image, dest_image, mask=face_mask)
        result = cv2.add(head_without_face, new_face)

        (x, y, w, h) = cv2.boundingRect(dest_convexHull)
        center_face = (int((x + x + w) / 2), int((y + y + h) / 2))

        return cv2.seamlessClone(result, dest_image, head_mask, center_face, cv2.MIXED_CLONE)

    def set_src_image(self,image):
        global src_image, src_image_gray, src_mask, src_landmark_points, src_np_points, src_convexHull, indexes_triangles
        src_image = image
        src_image_gray = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
        src_mask = np.zeros_like(src_image_gray)

        src_landmark_points = self.get_landmark_points(src_image)
        src_np_points = np.array(src_landmark_points)
        src_convexHull = cv2.convexHull(src_np_points)
        cv2.fillConvexPoly(src_mask, src_convexHull, 255)

        indexes_triangles = self.get_triangles(convexhull=src_convexHull,
                                                    landmarks_points=src_landmark_points,
                                                    np_points=src_np_points)

    def face_swapping(self,SRC_FILE,DEST_FILE):
        print(f"-------> This is SOURCE : {SRC_FILE}")
        print(f"-------> This is DESTINATION : {DEST_FILE}")

        src_image = cv2.imread(SRC_FILE)
        src_image_gray = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
        src_mask = np.zeros_like(src_image_gray)

        src_landmark_points = self.get_landmark_points(src_image)
        src_np_points = np.array(src_landmark_points)
        src_convexHull = cv2.convexHull(src_np_points)
        cv2.fillConvexPoly(src_mask, src_convexHull, 255)

        indexes_triangles = self.get_triangles(convexhull=src_convexHull, landmarks_points=src_landmark_points,
                                                    np_points=src_np_points)

        # Destination
        dest_image = cv2.imread(DEST_FILE)
        dest_image_gray = cv2.cvtColor(dest_image, cv2.COLOR_BGR2GRAY)
        dest_mask = np.zeros_like(dest_image_gray)

        dest_landmark_points = self.get_landmark_points(dest_image)
        dest_np_points = np.array(dest_landmark_points)
        dest_convexHull = cv2.convexHull(dest_np_points)

        height, width, channels = dest_image.shape
        new_face = np.zeros((height, width, channels), np.uint8)

        # Triangulation of both faces
        for triangle_index in indexes_triangles:
            # Triangulation of the first face
            points, cropped_triangle, cropped_triangle_mask, _ = self.triangulation(triangle_index=triangle_index,
                                                                                        landmark_points=src_landmark_points,
                                                                                        img=src_image)

            # Triangulation of second face
            points2, _, cropped_triangle_mask2, rect = self.triangulation(triangle_index=triangle_index,
                                                                                landmark_points=dest_landmark_points)

            # Warp triangles
            warped_triangle = self.warp_triangle(rect=rect, points1=points, points2=points2,
                                                        src_cropped_triangle=cropped_triangle,
                                                        dest_cropped_triangle_mask=cropped_triangle_mask2)

            # Reconstructing destination face
            self.add_piece_of_new_face(new_face=new_face, rect=rect, warped_triangle=warped_triangle)

        # Face swapped (putting 1st face into 2nd face)
        new_face = cv2.medianBlur(new_face, 3)
        result = self.swap_new_face(dest_image=dest_image, dest_image_gray=dest_image_gray,
                                        dest_convexHull=dest_convexHull, new_face=new_face)
        
        return result

    def vid_swapping(self, dest_image):
        dest_image_gray = cv2.cvtColor(dest_image, cv2.COLOR_BGR2GRAY)
        dest_mask = np.zeros_like(dest_image_gray)

        dest_landmark_points = self.get_landmark_points(dest_image)
        if dest_landmark_points is None:
            return
        dest_np_points = np.array(dest_landmark_points)
        dest_convexHull = cv2.convexHull(dest_np_points)

        height, width, channels = dest_image.shape
        new_face = np.zeros((height, width, channels), np.uint8)


        for triangle_index in indexes_triangles:

            points, src_cropped_triangle, cropped_triangle_mask, _ = self.triangulation(
                triangle_index=triangle_index,
                landmark_points=src_landmark_points,
                img=src_image)

            points2, _, dest_cropped_triangle_mask, rect = self.triangulation(triangle_index=triangle_index,
                                                                                    landmark_points=dest_landmark_points)

            warped_triangle = self.warp_triangle(rect=rect, points1=points, points2=points2,
                                                        src_cropped_triangle=src_cropped_triangle,
                                                        dest_cropped_triangle_mask=dest_cropped_triangle_mask)

            self.add_piece_of_new_face(new_face=new_face, rect=rect, warped_triangle=warped_triangle)
        result = self.swap_new_face(dest_image=dest_image, dest_image_gray=dest_image_gray,
                                        dest_convexHull=dest_convexHull, new_face=new_face)
        
        return result
    

class BasicApp(App):
    img_link = "C:\\Users\\rohai\\OneDrive\\Desktop\\College\\DIP\\DIPproj\\Final_Proj\\images\\bill_gates.jpg"
    image = cv2.imread(img_link)

    SwapObj = FaceSwap()

    SwapObj.set_src_image(image)

    def __init__(self):
        Clock.schedule_interval(self.update, 1.0/33.0)        

    # def build(self):
    #     self.web_cam = Image(size_hint = (1,0.8), source = self.img_link)
    #     self.web_cam_2 = Image(size_hint = (1,0.8))                     # Second Camera Feed
    #     self.button = Button(text = "Upload Image", size_hint = (1,.1), on_press = self.show_file_chooser)

    #     cam_layout = BoxLayout(orientation='horizontal')        
    #     cam_layout.add_widget(self.web_cam)
    #     # layout.add_widget(self.button)
    #     cam_layout.add_widget(self.web_cam_2)

    #     master_layout = BoxLayout(orientation = "vertical")
    #     master_layout.add_widget(cam_layout)
    #     master_layout.add_widget(self.button)

    #     self.capture = cv2.VideoCapture(0)
    #     Clock.schedule_interval(self.update, 1.0/33.0)

    #     return master_layout

    def show_file_chooser(self,instance):
        file_chooser = FileChooserListView()
        file_chooser.bind(on_submit=self.load_image)
        self.root.add_widget(file_chooser)

    def load_image(self, instance, _, __):
        file_path = instance.selection[0]
        print(file_path)
        # self.image.source = file_path
        self.image = cv2.imread(file_path)
        self.SwapObj.set_src_image(self.image)
        self.root.remove_widget(instance)

    def update(self, *args):
        global src_image, src_image_gray, src_mask, src_landmark_points, src_np_points, src_convexHull, indexes_triangles
        try:
            ret, frame = self.capture.read()
            # frame = frame[120:120+250, 200:200+250, :]

            dest_image = frame

            # Original Video Feed
            buf = cv2.flip(frame, 0).tobytes()
            img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.web_cam.texture = img_texture

            # Face Swapped Feed
            result = self.SwapObj.vid_swapping(dest_image)

            result = cv2.medianBlur(result, 3)
        except:
            result = frame

        buf = cv2.flip(result, 0).tobytes()
        img_texture = Texture.create(size=(result.shape[1], result.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.ids.vidfeed.texture = img_texture   # Second Camera Feed


class VidSwap(Screen):
    img_link = "C:\\Users\\rohai\\OneDrive\\Desktop\\College\\DIP\\DIPproj\\Final_Proj\\images\\bill_gates.jpg"
    image = cv2.imread(img_link)

    SwapObj = FaceSwap()

    SwapObj.set_src_image(image)

    def __init__(self, **kwargs):
        super(VidSwap, self).__init__(**kwargs)

        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)        

    # def build(self):
    #     self.web_cam = Image(size_hint = (1,0.8), source = self.img_link)
    #     self.web_cam_2 = Image(size_hint = (1,0.8))                     # Second Camera Feed
    #     self.button = Button(text = "Upload Image", size_hint = (1,.1), on_press = self.show_file_chooser)

    #     cam_layout = BoxLayout(orientation='horizontal')        
    #     cam_layout.add_widget(self.web_cam)
    #     # layout.add_widget(self.button)
    #     cam_layout.add_widget(self.web_cam_2)

    #     master_layout = BoxLayout(orientation = "vertical")
    #     master_layout.add_widget(cam_layout)
    #     master_layout.add_widget(self.button)

    #     self.capture = cv2.VideoCapture(0)
    #     Clock.schedule_interval(self.update, 1.0/33.0)

    #     return master_layout

    def file_chooser(self):
        filechooser.open_file(on_selection = self.selected)

    def selected(self,selection):
        if selection:
            self.ids.img.source = selection[0].replace("\\","\\\\")
            self.image = cv2.imread(self.ids.img.source)

            self.SwapObj.set_src_image(self.image)

    def update(self, *args):
        global src_image, src_image_gray, src_mask, src_landmark_points, src_np_points, src_convexHull, indexes_triangles
        
        ret, frame = self.capture.read()
        result = frame
            # frame = frame[120:120+250, 200:200+250, :]
        try:
            dest_image = frame

            # Face Swapped Feed
            result = self.SwapObj.vid_swapping(dest_image)

            result = cv2.medianBlur(result, 3)
        except:
            result = frame

        buf = cv2.flip(result, 0).tobytes()
        img_texture = Texture.create(size=(result.shape[1], result.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.ids.vidfeed.texture = img_texture   # Second Camera Feed


class PicSwap(Screen):
    img1 = "C:\\Users\\rohai\\OneDrive\\Desktop\\College\\DIP\\DIPproj\\Final_Proj\\images\\bill_gates.jpg"
    img2 = "C:\\Users\\rohai\\OneDrive\\Desktop\\College\\DIP\\DIPproj\\Final_Proj\\images\\dwayne_johnson.jpg"

    orig_texture_1 = None
    orig_texture_2 = None

    swapObj = FaceSwap()

    # def show_file_chooser(self,instance):
    #     print("FUNCTION EXECUTED")
    #     file_chooser = FileChooserListView()
    #     file_chooser.bind(on_submit=self.load_image)
    #     self.root.add_widget(file_chooser)

    # def load_image(self, instance, _, __):
    #     file_path = instance.selection[0]
    #     print(file_path)
    #     # self.image.source = file_path
    #     self.image = cv2.imread(file_path)
    #     self.SwapObj.set_src_image(self.image)
    #     self.root.remove_widget(instance)

    def update_result(self):
        res1 = self.swapObj.face_swapping(self.ids.img1.source,self.ids.img2.source)
        res2 = self.swapObj.face_swapping(self.ids.img2.source,self.ids.img1.source)

        self.orig_texture_1 = self.ids.img1.texture
        self.orig_texture_2 = self.ids.img2.texture

        buf = cv2.flip(res1, 0).tobytes()
        img_texture = Texture.create(size=(res1.shape[1], res1.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

        self.ids.img2.texture = img_texture

        buf = cv2.flip(res2, 0).tobytes()
        img_texture = Texture.create(size=(res2.shape[1], res2.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

        self.ids.img1.texture = img_texture

        print(f"This is TEXTURE 1 : {self.orig_texture_1}")
        print(f"This is TEXTURE 2 : {self.orig_texture_2}")


    def file_chooser_1(self):
        filechooser.open_file(on_selection = self.selected1)

    def selected1(self,selection):
        if selection:
            self.ids.img1.source = selection[0].replace("\\","\\\\")
            self.orig_texture_1 = self.ids.img1.texture
        if self.orig_texture_2 is not None:
            self.ids.img2.texture = self.orig_texture_2
            self.orig_texture_2 = None

        print(f"This is TEXTURE 1 : {self.orig_texture_1}")
        print(f"This is TEXTURE 2 : {self.orig_texture_2}")

    def file_chooser_2(self):
        filechooser.open_file(on_selection = self.selected2)

    def selected2(self,selection):
        if selection:
            self.ids.img2.source = selection[0].replace("\\","\\\\")
            self.orig_texture_2 = self.ids.img2.texture
        if self.orig_texture_1 is not None:
            self.ids.img1.texture = self.orig_texture_1
            self.orig_texture_1 = None

        print(f"This is TEXTURE 1 : {self.orig_texture_1}")
        print(f"This is TEXTURE 2 : {self.orig_texture_2}")

kv = Builder.load_file("main.kv")    
class FinalApp(App):
    def build(self):
        return kv

if __name__ == '__main__':
    FinalApp().run()