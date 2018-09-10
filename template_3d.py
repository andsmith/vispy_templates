import pprint
import numpy as np

import vispy.app
import vispy.scene.visuals
from vispy.scene.cameras import TurntableCamera
from vispy.util.quaternion import Quaternion



class VispyTest(object):
    def __init__(self):


        self._vispy_init()

        self._axes_points = []
        for i in range(-1,2,2):
            for j in range(-1,2,2):
                for k in range(-1,2,2):
                    #if i==k==j==0:
                    #    continue
                    self._axes_points.append([float(i), float(j), float(k)])
        self._axes_points= np.array(self._axes_points)
        pprint.pprint(self._axes_points)

        self.add_points(self._axes_points, [.6, .6, .6], [.8, .8, .8], 10, 'axis_points')
        self.add_xy_rect(np.zeros(3) , 1.5, .5, [.55, .77, 0.99, .8], [1.0, 1.0, 1.0, 0.9])

    def start(self):
        vispy.app.run()


    def _vispy_init(self):

        self._points = {}
        self._xyrects = {}
        self._polygons = {}
        self._canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, title="vispy template")
        self._view = self._canvas.central_widget.add_view()
        self._view.camera = TurntableCamera(parent=self._view.scene)  # NoAccelerationFlyCamera(speed_scaler=0.5)
        self._debug_axis = vispy.scene.visuals.XYZAxis(parent=self._view.scene)


    def add_points(self, coords, color, edge_color, size, name=None):
        name = int(np.random.random_integers(0, 65536, 1))
        self._points[name] = vispy.scene.visuals.Markers(parent=self._view.scene)
        self._points[name].set_data(pos=coords, face_color=color, edge_color=edge_color, size=size)
        self._points[name].visible=True
        #self._points[name].update()


    def add_xy_rect(self, center, height, width, color, border_color, radius = 0.2, name=None):
        name = int(np.random.random_integers(0, 65536, 1))
        self._xyrects[name] = vispy.scene.visuals.Rectangle(parent=self._view.scene, center = center, height=height, width=width, color=color, border_color=border_color, radius=radius)
        self._xyrects[name].visible=True



    def add_polygon(self, coords, color, border_color, radius = 0.2, name=None):
        name = int(np.random.random_integers(0, 65536, 1))
        self._polygons[name] = vispy.scene.visuals.Polygon(parent=self._view.scene, pos=coords, color=color, border_color=border_color, radius=radius)
        self._polygons[name].visible=True


def minimal_vispy_function():

    pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    canv = vispy.scene.SceneCanvas(show=True)
    view = canv.central_widget.add_view()
    view.camera = vispy.scene.cameras.TurntableCamera(parent=view.scene)
    marks = vispy.scene.visuals.Markers(pos=pts, size=5, face_color=[1.0, 1.0, 1.0], edge_color = [1.0, 0.2, 0.2], parent=view.scene)
    #view.add(marks)
    marks.visible = True

    #marks.visible=True
    vispy.app.run()

if __name__ == "__main__":
    #minimal_vispy_function()
    vt = VispyTest()
    vt.start()