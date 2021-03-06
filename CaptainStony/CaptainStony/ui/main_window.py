import sys
import numpy as np
import cv2
import configparser

from PyQt5 import QtGui, QtWidgets
from opendr.camera import ProjectPoints, Rodrigues
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight

from .gen.main_window import Ui_MainWindow as Ui_MainWindow_Base
from .camera_widget import Ui_CameraWidget
from .util import *

from PoseRetriever import *
from WebcamWidget import *


model_type_list = ['smplx','smpl','flame']

class Ui_MainWindow(QtWidgets.QMainWindow, Ui_MainWindow_Base):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)

        self._moving = False
        self._rotating = False
        self._mouse_begin_pos = None
        self._update_canvas = False

        self.camera = ProjectPoints(rt=np.zeros(3), t=np.zeros(3))
        self.joints2d = ProjectPoints(rt=np.zeros(3), t=np.zeros(3))
        self.frustum = {'near': 0.1, 'far': 1000., 'width': 100, 'height': 30}
        self.light = LambertianPointLight(vc=np.array([0.98, 0.98, 0.98]), light_color=np.array([1., 1., 1.]))
        self.rn = ColoredRenderer()
        self.rn.set(glMode='glfw',bgcolor=np.array([151 / 255, 102 / 255, 10 / 255]), frustum=self.frustum, camera=self.camera, vc=self.light,
                                  overdraw=True, msaa = True, nsamples = 8, sharedWin = None)
        #self.rn.initGL()
        self.rn.debug = False

        self.model_type = 'smpl'
        self.model_gender = 'm'
        self.model = None
        self._init_model()
        self.model.pose[0] = np.pi

        self.camera_widget = Ui_CameraWidget(self.camera, self.frustum, self.draw)
        self.btn_camera.clicked.connect(lambda: self._show_camera_widget())

        self.toggle_active_pose_panel(active = True)

        self.pos_0.valueChanged[float].connect(lambda val: self._update_position(0, val))
        self.pos_1.valueChanged[float].connect(lambda val: self._update_position(1, val))
        self.pos_2.valueChanged[float].connect(lambda val: self._update_position(2, val))

        #self.radio_f.pressed.connect(lambda: self._init_model('f'))
        #self.radio_m.pressed.connect(lambda: self._init_model('m'))

        self.reset_pose.clicked.connect(self._reset_pose)
        self.reset_shape.clicked.connect(self._reset_shape)
        #self.reset_expression.clicked.connect(self._reset_expression)
        self.reset_postion.clicked.connect(self._reset_position)

        self.canvas.wheelEvent = self._zoom
        self.canvas.mousePressEvent = self._mouse_begin
        self.canvas.mouseMoveEvent = self._move
        self.canvas.mouseReleaseEvent = self._mouse_end

        self.action_save.triggered.connect(self._save_config_dialog)
        self.action_open.triggered.connect(self._open_config_dialog)
        self.action_save_screenshot.triggered.connect(self._save_screenshot_dialog)
        self.action_save_mesh.triggered.connect(self._save_mesh_dialog)

        self.view_joints.triggered.connect(self.draw)
        self.view_joint_ids.triggered.connect(self.draw)
        self.view_bones.triggered.connect(self.draw)

        self._update_canvas = True

        self.initialize_pose_retrieval_widgets()

        self.webcam = WebcamControlWindow()
        self.groupBox.layout().addWidget(self.webcam, 2)


    def showEvent(self, event):
        self._init_camera()
        super(self.__class__, self).showEvent(event)

    def resizeEvent(self, event):
        self._init_camera()
        super(self.__class__, self).resizeEvent(event)

    def closeEvent(self, event):
        self.camera_widget.close()
        super(self.__class__, self).closeEvent(event)

    def draw(self):
        # print(self._update_canvas)
        if self._update_canvas:
            # print(self.model.r)
            img = np.array(self.rn.r)

            if self.view_joints.isChecked() or self.view_joint_ids.isChecked() or self.view_bones.isChecked():
                img = self._draw_annotations(img)

            self.canvas.setScaledContents(False)
            self.canvas.setPixmap(self._to_pixmap(img))
            self.canvas.setScaledContents(True)
            self.canvas_query.setPixmap(self._to_pixmap(img))
            self._update_pose_text_edit()

    def _draw_annotations(self, img):
        self.joints2d.set(t=self.camera.t, rt=self.camera.rt, f=self.camera.f, c=self.camera.c, k=self.camera.k)

        height = self.canvas.height()
        if self.view_bones.isChecked():
            kintree = self.model.kintree_table[:, 1:]
            for k in range(kintree.shape[1]):
                cv2.line(img, (int(self.joints2d.r[kintree[0, k], 0]), int(height - self.joints2d.r[kintree[0, k], 1])),
                         (int(self.joints2d.r[kintree[1, k], 0]), int(height - self.joints2d.r[kintree[1, k], 1])),
                         (0.98, 0.98, 0.98), 3)

        if self.view_joints.isChecked():
            for j in self.joints2d.r:
                jj = height-j[1] # for opengl: flipx
                cv2.circle(img, (int(j[0]), int(jj)), 5, (0.38, 0.68, 0.15), -1)

        if self.view_joint_ids.isChecked():
            for k, j in enumerate(self.joints2d.r):
                jj = height-j[1]
                cv2.putText(img, str(k), (int(j[0]), int(jj)), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0.3, 0.23, 0.9), 2)

        return img

    def _init_model(self, gender=None):
        pose = None
        betas = None
        trans = None

        if self.model is not None:
            pose = self.model.pose.r
            betas = self.model.betas.r
            trans = self.model.trans.r

        if gender == None: 
            gender = self.model_gender
        else:
            self.model_gender = gender
        
        self.model = load_model(model_type=self.model_type, gender=self.model_gender)
        # print(self.model.r)

        if pose is not None:
            self.model.pose[:] = pose
            self.model.betas[:] = betas
            self.model.trans[:] = trans

        self.light.set(v=self.model, f=self.model.f, num_verts=len(self.model))
        self.rn.set(v=self.model, f=self.model.f)

        self.camera.set(v=self.model)
        self.joints2d.set(v=self.model.J_transformed)

        self.draw()

    def _init_camera(self, update_camera=False):
        w = self.canvas.width()
        h = self.canvas.height()

        if update_camera or w != self.frustum['width'] and h != self.frustum['height']:
            self.camera.set(rt=np.array([self.camera_widget.rot_0.value(), self.camera_widget.rot_1.value(),
                                         self.camera_widget.rot_2.value()]),
                            t=np.array([self.camera_widget.pos_0.value(), self.camera_widget.pos_1.value(),
                                        self.camera_widget.pos_2.value()]),
                            f=np.array([w, w]) * self.camera_widget.focal_len.value(),
                            c=np.array([w, h]) / 2.,
                            k=np.array([self.camera_widget.dist_0.value(), self.camera_widget.dist_1.value(),
                                        self.camera_widget.dist_2.value(), self.camera_widget.dist_3.value(),
                                        self.camera_widget.dist_4.value()]))

            self.frustum['width'] = w
            self.frustum['height'] = h

            self.light.set(light_pos=Rodrigues(self.camera.rt).T.dot(self.camera.t) * -10.)
            self.rn.set(frustum=self.frustum, camera=self.camera)
            flipXRotation = np.array([[1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0., 0.0],
                [0.0, 0., -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]])
            self.rn.camera.openglMat = flipXRotation #this is from setupcamera in utils
            self.rn.initGL()           

            self.draw()

    def _save_config_dialog(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save config', None, 'Config File (*.ini)')
        if filename:
            with open(str(filename), 'w') as fp:
                config = configparser.ConfigParser()
                config.add_section('Model')
                config.set('Model', 'gender', self.model_gender)
                config.set('Model', 'shape', ','.join(str(s) for s in self.model.betas.r))
                config.set('Model', 'pose', ','.join(str(p) for p in self.model.pose.r))
                config.set('Model', 'translation', ','.join(str(p) for p in self.model.trans.r))

                config.add_section('Camera')
                config.set('Camera', 'translation', ','.join(str(t) for t in self.camera.t.r))
                config.set('Camera', 'rotation', ','.join(str(r) for r in self.camera.rt.r))
                config.set('Camera', 'focal_length', str(self.camera_widget.focal_len.value()))
                config.set('Camera', 'center', '{},{}'.format(self.camera_widget.center_0.value(),
                                                              self.camera_widget.center_1.value()))
                config.set('Camera', 'distortion', ','.join(str(r) for r in self.camera.k.r))

                config.write(fp)

    def _open_config_dialog(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Load config', None, 'Config File (*.ini)')
        if filename:
            config = configparser.ConfigParser()
            config.read(str(filename))

            self._update_canvas = False
            self._init_model(config.get('Model', 'gender'))

            shapes = np.fromstring(config.get('Model', 'shape'), dtype=np.float64, sep=',')
            poses = np.fromstring(config.get('Model', 'pose'), dtype=np.float64, sep=',')
            position = np.fromstring(config.get('Model', 'translation'), dtype=np.float64, sep=',')

            for key, shape in self._shapes():
                val = shapes[key] / 5.0 * 50.0 + 50.0
                shape.setValue(val)
            for key, pose in self._poses():
                if key == 0:
                    val = (poses[key] - np.pi) / np.pi * 50.0 + 50.0
                else:
                    val = poses[key] / np.pi * 50.0 + 50.0
                pose.setValue(val)

            self.pos_0.setValue(position[0])
            self.pos_1.setValue(position[1])
            self.pos_2.setValue(position[2])

            cam_pos = np.fromstring(config.get('Camera', 'translation'), dtype=np.float64, sep=',')
            cam_rot = np.fromstring(config.get('Camera', 'rotation'), dtype=np.float64, sep=',')
            cam_dist = np.fromstring(config.get('Camera', 'distortion'), dtype=np.float64, sep=',')
            cam_c = np.fromstring(config.get('Camera', 'center'), dtype=np.float64, sep=',')
            cam_f = config.getfloat('Camera', 'focal_length')
            print(cam_c)
            self.camera_widget.set_values(cam_pos, cam_rot, cam_f, cam_c, cam_dist)

            self._update_canvas = True
            self.draw()

    def _save_screenshot_dialog(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save screenshot', None, 'Images (*.png *.jpg *.ppm)')
        if filename:
            img = np.array(self.rn.r)
            if self.view_joints.isChecked() or self.view_joint_ids.isChecked() or self.view_bones.isChecked():
                img = self._draw_annotations(img)
            cv2.imwrite(str(filename), np.uint8(img * 255))

    def _save_mesh_dialog(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save mesh', None, 'Mesh (*.obj)')
        if filename:
            with open(filename, 'w') as fp:
                for v in self.model.r:
                    fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

                for f in self.model.f + 1:
                    fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    def _zoom(self, event):
        delta = -event.angleDelta().y() / 1200.0
        self.camera_widget.pos_2.setValue(self.camera_widget.pos_2.value() + delta)

    def _mouse_begin(self, event):
        if event.button() == 4: # middle
            self._moving = True
        elif event.button() == 1: # left
            self._rotating = True
        self._mouse_begin_pos = event.pos()

    def _mouse_end(self, event):
        self._moving = False
        self._rotating = False

    def _move(self, event):
        if self._moving:
            delta = event.pos() - self._mouse_begin_pos
            self.camera_widget.pos_0.setValue(self.camera_widget.pos_0.value() + delta.x() / 500.)
            self.camera_widget.pos_1.setValue(self.camera_widget.pos_1.value() + delta.y() / 500.)
            self._mouse_begin_pos = event.pos()
        elif self._rotating:
            delta = event.pos() - self._mouse_begin_pos
            self.camera_widget.rot_0.setValue(self.camera_widget.rot_0.value() + delta.y() / 300.)
            self.camera_widget.rot_1.setValue(self.camera_widget.rot_1.value() - delta.x() / 300.)
            self._mouse_begin_pos = event.pos()

    def _show_camera_widget(self):
        self.camera_widget.show()
        self.camera_widget.raise_()

    def _update_shape(self, id, val):
        val = (val - 50) / 50.0 * 5.0
        self.model.betas[id] = val
        self.draw()

    def _update_exp(self, id, val):
        val = (val - 50) / 50.0 * 5.0
        if self.model_type=='smplx':
            self.model.betas[10 + id] = val
            self.draw()
        elif self.model_type=='flame':
            self.model.betas[10 + id] = val
            self.draw()

    def _reset_shape(self):
        self._update_canvas = False
        for key, shape in self._shapes():
            shape.setValue(50)
        self._update_canvas = True
        self.draw()

    def _reset_expression(self):
        if self.model_type != 'smplx' and self.model_type != 'flame':
            return
        self._update_canvas = False
        for key, exp in self._expressions():
            exp.setValue(50)
        self._update_canvas = True
        self.draw()

    @staticmethod
    def _convert_scrollbar_value(id, val):
        val = (val - 50) / 50.0 * np.pi

        #if id == 0:
        #    val += np.pi

        return val

    def _update_pose(self, id, val):
        val = self._convert_scrollbar_value(id, val)

        if self.model_type == 'flame' and id>=5*3:
            return

        self.model.pose[id] = val
        self.draw()

    def _reset_pose(self):
        self._update_canvas = False
        for key, pose in self._poses():
            pose.setValue(50)
        self._update_canvas = True
        self.draw()

    def _update_position(self, id, val):
        self.model.trans[id] = val
        self.draw()

    ################## Pose retrieval task ##################
    def toggle_active_pose_panel(self, active = True):

        if active:
            for key, shape in self._shapes():
                shape.valueChanged[int].connect(lambda val, k=key: self._update_shape(k, val))

            for key, exp in self._expressions():
                exp.valueChanged[int].connect(lambda val, k=key: self._update_exp(k, val))

            for key, pose in self._poses():
                pose.valueChanged[int].connect(lambda val, k=key: self._update_pose(k, val))
        else:
            for key, shape in self._shapes():
                shape.valueChanged[int].disconnect()

            for key, exp in self._expressions():
                exp.valueChanged[int].disconnect()

            for key, pose in self._poses():
                pose.valueChanged[int].disconnect()


    def initialize_pose_retrieval_widgets(self):

        self.canvas_query.wheelEvent = self._zoom
        self.canvas_query.mousePressEvent = self._mouse_begin
        self.canvas_query.mouseMoveEvent = self._move
        self.canvas_query.mouseReleaseEvent = self._mouse_end

        # main parts
        self.btn_query.clicked.connect(lambda _ : self._query_current_pose())
        self.check_load_pose.stateChanged.connect(lambda _: self._toggle_auto_load())
        self.btn_load.clicked.connect(lambda _ : self._load_selected_pose())
        self.data_list.currentItemChanged.connect(lambda _: self._update_selected_data_view())
        self.result_list.currentItemChanged.connect(lambda _: self._update_selected_view())
        self._toggle_auto_load()

        # radio buttons of metrics
        self.radio_l1.clicked.connect(lambda _: self._set_distance_type())
        self.radio_l2.clicked.connect(lambda _: self._set_distance_type())
        self.radio_canberra.clicked.connect(lambda _: self._set_distance_type())
        self.radio_chebyshev.clicked.connect(lambda _: self._set_distance_type())
        self.radio_cosine.clicked.connect(lambda _: self._set_distance_type())

        # radio buttons of masks
        self.radio_use_all.clicked.connect(lambda _: self._set_mask_type())
        self.radio_ignore_far.clicked.connect(lambda _: self._set_mask_type())
        self.radio_use_trunk.clicked.connect(lambda _: self._set_mask_type())



        self.large_memory_mode = False
        self.pose_retriever = PoseRetriever(load_image_into_memory = self.large_memory_mode)
        data_path = os.environ["SMPL_VIEWER_DATA_DIR"]
        self.pose_retriever.Load3DPWProcessedData(data_path)
        self._update_data_list()
        
    def _get_pose_parameters(self):
        #return [ self._convert_scrollbar_value(idx, wgt.value()) for idx, wgt in list(self._poses()) ]
        return [ self.model.pose.r[i] for i in range(len(self.model.pose))]


    def _load_selected_pose(self):
        print("Loading selected pose")
        if self.data_list.currentItem() is None:
            return

        image_path = self.data_list.currentItem().text()
        pose = self.pose_retriever.GetPose(image_path)
        self.model.pose[3:] = pose[3:]
        self._update_pose_panel()
        self.draw()


    def _update_pose_panel(self):
        self.toggle_active_pose_panel(active = False)
        slider_values = [ round(self.model.pose.r[i] / np.pi * 50 + 50) for i in range(72)]

        for key, pose in self._poses():
            if key < 3:
                continue
            pose.setValue(slider_values[key])

        self.toggle_active_pose_panel(active = True)

    def _update_pose_text_edit(self):
        p = self._get_pose_parameters()
        text_p = "\t".join([ "{:.4f}".format(d) for d in p ])
        self.pose_text_edit.setText(text_p)

     
    def _query_current_pose(self):
        print("Query start")
        pose_param = self._get_pose_parameters()
        query_results = self.pose_retriever.Query(pose = pose_param)
        self._update_result_list(query_results)
        self._update_selected_view()


    def _update_data_list(self):
        data_list = self.pose_retriever.GetDataList()

        for d in data_list:
            li = QtWidgets.QListWidgetItem(parent = self.data_list)
            li.setText(d)
            self.data_list.addItem(li)
        
        ## set first as selected, if at least one exists
        #if self.data_list.count() > 0:
        #    self.data_list.setCurrentRow(0)

    def _update_result_list(self, query_results):
        print("Updating result list")
        self.result_list.clear()
        for qr in query_results:
            li = QtWidgets.QListWidgetItem(parent = self.result_list)
            li.setText(qr[0])
            self.result_list.addItem(li)
        
        # set first as selected, if at least one exists
        if self.result_list.count() > 0:
            self.result_list.setCurrentRow(0)
        
    def _update_selected_data_view(self):
        print("Updating data view")
        if self.data_list.currentItem() is None:
            return

        key = self.data_list.currentItem().text()
        #self.pose_retriever.image_dict[image_path] = image_path
        selected_image = self.pose_retriever.GetImage(key)
        self.selected_data_view.setPixmap(
            self._to_pixmap(
                #self._get_transformed_display_image(selected_image),
                selected_image,
                bgr_to_rgb = False,
                resize = (800, 400)))

    def _update_selected_view(self):
        print("Updating selected view")
        if self.result_list.currentItem() is None:
            return

        key = self.result_list.currentItem().text()
        #self.pose_retriever.image_dict[image_path] = image_path
        selected_image = self.pose_retriever.GetImage(key)
        self.selected_view.setPixmap(
            self._to_pixmap(
                #self._get_transformed_display_image(selected_image),
                selected_image,
                bgr_to_rgb = False,
                resize = (800, 400)))
        

    def _get_transformed_display_image(self, image):
        # split into two horizontally, then vstack;
        splitted = np.hsplit(image, 2)
        return np.vstack((splitted[0], splitted[1]))
        

    def _toggle_auto_load(self):
        if self.check_load_pose.isChecked():
            self.data_list.currentItemChanged.connect(lambda _: self._load_selected_pose())
        else:
            self.data_list.currentItemChanged.disconnect()
            self.data_list.currentItemChanged.connect(lambda _: self._update_selected_data_view())


    def _set_distance_type(self):
        if self.radio_l1.isChecked():
            distance_type = DISTANCE_L1_NORM
        elif self.radio_l2.isChecked():
            distance_type = DISTANCE_L2_NORM
        elif self.radio_canberra.isChecked():
            distance_type = DISTANCE_CANBERRA
        elif self.radio_chebyshev.isChecked():
            distance_type = DISTANCE_CHEBYSHEV
        elif self.radio_cosine.isChecked():
            distance_type = DISTANCE_COSINE

        self.pose_retriever.SetDistanceType(distance_type)
        logging.info("distance type set to {}".format(distance_type))
        self._query_current_pose()

    def _set_mask_type(self):
        if self.radio_use_all.isChecked():
            mask_type = MASK_USE_ALL
        elif self.radio_ignore_far.isChecked():
            mask_type = MASK_IGNORE_FAR_END
        elif self.radio_use_trunk.isChecked():
            mask_type = MASK_USE_TRUNK_ONLY

        self.pose_retriever.SetMaskType(mask_type)
        logging.info("mask type set to {}".format(mask_type))
        self._query_current_pose()


    ################## ##################

    def _reset_position(self):
        self._update_canvas = False
        self.pos_0.setValue(0)
        self.pos_1.setValue(0)
        self.pos_2.setValue(0)
        self._update_canvas = True
        self.draw()

    def _poses(self):
        return enumerate([
            self.pose_0,
            self.pose_1,
            self.pose_2,
            self.pose_3,
            self.pose_4,
            self.pose_5,
            self.pose_6,
            self.pose_7,
            self.pose_8,
            self.pose_9,
            self.pose_10,
            self.pose_11,
            self.pose_12,
            self.pose_13,
            self.pose_14,
            self.pose_15,
            self.pose_16,
            self.pose_17,
            self.pose_18,
            self.pose_19,
            self.pose_20,
            self.pose_21,
            self.pose_22,
            self.pose_23,
            self.pose_24,
            self.pose_25,
            self.pose_26,
            self.pose_27,
            self.pose_28,
            self.pose_29,
            self.pose_30,
            self.pose_31,
            self.pose_32,
            self.pose_33,
            self.pose_34,
            self.pose_35,
            self.pose_36,
            self.pose_37,
            self.pose_38,
            self.pose_39,
            self.pose_40,
            self.pose_41,
            self.pose_42,
            self.pose_43,
            self.pose_44,
            self.pose_45,
            self.pose_46,
            self.pose_47,
            self.pose_48,
            self.pose_49,
            self.pose_50,
            self.pose_51,
            self.pose_52,
            self.pose_53,
            self.pose_54,
            self.pose_55,
            self.pose_56,
            self.pose_57,
            self.pose_58,
            self.pose_59,
            self.pose_60,
            self.pose_61,
            self.pose_62,
            self.pose_63,
            self.pose_64,
            self.pose_65,
            self.pose_66,
            self.pose_67,
            self.pose_68,
            self.pose_69,
            self.pose_70,
            self.pose_71,
        ])

    def _shapes(self):
        return enumerate([
            self.shape_0,
            self.shape_1,
            self.shape_2,
            self.shape_3,
            self.shape_4,
            self.shape_5,
            self.shape_6,
            self.shape_7,
            self.shape_8,
            self.shape_9,
        ])

    def _expressions(self):
        return enumerate([
            self.shape_10,
            self.shape_11,
            self.shape_12,
            self.shape_13,
            self.shape_14,
            self.shape_15,
            self.shape_16,
            self.shape_17,
            self.shape_18,
            self.shape_19,
        ])

    @staticmethod
    def _to_pixmap(im, bgr_to_rgb = True, resize = None):
        if im.dtype == np.float32 or im.dtype == np.float64:
            im = np.uint8(im * 255)

        if len(im.shape) < 3 or im.shape[-1] == 1:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        elif bgr_to_rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        if resize is not None:
            im = cv2.resize(im, dsize = resize)

        qimg = QtGui.QImage(im, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888)

        return QtGui.QPixmap(qimg)
