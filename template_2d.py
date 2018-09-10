from copy import deepcopy
import os
import logging
import argparse
import vispy.scene
import sys
import subprocess32 as subprocess
import numpy as np
import vispy.scene
import vispy.app
from vispy.scene.visuals import Image
from vispy.visuals.transforms import STTransform
from threading import Lock

# ##### GUI DEFINITIONS:  colors, drawing styles, modes, etc.
# image colors use 0-255


class VispyUI(object):
    """
    Display an image, rectangle, text, enable pan/tilt, keyboard
    """
    def __init__(self):
        self._make_new_random_image()
        # state
        self._last_mouse_pos = np.array([0.0, 0.0])
        self._load_data()  # Load data in after the view has been setup
        self._canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, title="Route editor -- H for help")
        self._viewbox = self._canvas.central_widget.add_view()
        self._viewbox.camera = vispy.scene.cameras.PanZoomCamera(parent=self._viewbox.scene, aspect=1)
        self._add_graphics_elements()

        self._canvas.events.key_press.connect(self._on_key_press)
        self._canvas.events.mouse_move.connect(self._on_mouse_move)
        self._canvas.events.mouse_press.connect(self._on_mouse_press)
        self._canvas.events.mouse_release.connect(self._on_mouse_release)

        self._set_map_image(self._image)

    def _make_new_random_image(self):
        self._h, self._w = 100 + int(np.random.rand(1) * 100), 100 + int(np.random.rand(1) * 100)
        self._image = np.uint8(np.zeros((self._h,self._w, 3)) + 255 * np.random.rand(self._h*self._w*3).reshape((self._h,self._w, 3)))


    def _add_graphics_elements(self):
        """
        Create all the graphics objects (VISPY objects), put them on the canvas.
        """

        # Image
        self._image_object = Image(self._image, parent=self._viewbox.scene)
        self._image_object.set_gl_state('translucent', depth_test=False)
        self._image_object.order = 1
        self._image_object.visible = True

        self._text_box_width = 150
        self._text_box_height = 40
        self._text_box_offset = 10
        # Text background box in upper-left corner
        self._text_bkg_rect = vispy.scene.visuals.Rectangle([ self._text_box_width/2+self._text_box_offset,
                                                              self._text_box_height/2+self._text_box_offset],
                                                                     color=[0.1, 0.0, 0.0,.8],
                                                                     border_color=[0.1, 0.0, 0.0],
                                                                     border_width=2,
                                                                     height=self._text_box_height,
                                                                     width=self._text_box_width,
                                                                     radius=10.0,
                                                                     parent=self._canvas.scene)
        self._text_bkg_rect.set_gl_state('translucent', depth_test=False)
        self._text_bkg_rect.visible = True
        self._text_bkg_rect.order=2

        # Text
        self._text = "?"
        self._text_pos = [self._text_box_offset + 10, self._text_box_offset+10]
        self._text_obj = vispy.scene.visuals.Text(self._text,
                                                     parent=self._canvas.scene,
                                                     color=[0.9, 0.8, 0.8],
                                                     anchor_x='left',
                                                     anchor_y='top')
        self._text_obj.pos= self._text_pos
        self._text_obj.font_size = 18
        self._text_obj.visible = True
        self._text_obj.order = 3

    def _change_text(self, new_text):
        if new_text is not None:
            self._text = new_text
        else:
            self._text = ""
        self._text_obj.text = self._text


    def _load_data(self):
        """
        Downlaod (if necessary) and load into memory the costmap, route and scrubber deck state controls.
        """
        self._images = []


    def _set_map_image(self, disp_image=None):
        """
        Extract an image to display from the current cost map.
        """
        if disp_image is not None:
            self._image = disp_image
        xmin, ymin = 0, 0
        xmax, ymax = self._image.shape[1], self._image.shape[0]
        self._viewbox.camera.set_range(x=(xmin, xmax), y=(ymin, ymax), z=None)
        self._image_object.set_data(self._image)
        self._image_object.visible = True

    def _on_key_press(self, ev):
        """
        vispy keyboard callback
        """

        if not ev or not ev.key:
            # on Mac sometimes this callback is triggered with 
            # NoneType key when clicking between desktops or full screen apps
            return

        control_pressed = 'Control' in [e.name for e in ev.modifiers]
        shift_pressed = 'Shift' in [e.name for e in ev.modifiers]
        print "Shift:  %s\nControl:%s" % (shift_pressed, control_pressed)
        if ev.key.name == "Space":

            self._make_new_random_image()
            self._set_map_image()

        elif ev.key.name == "Z":
            print "Z"
        elif ev.key.name == "M":
            print "M"

        else:
            self._change_text(ev.key.name)
            logging.info("Unknown keypress:  %s" % (ev.key.name, ))

    def _on_mouse_move(self, event):
        """
        vispy mouse motion callback
        All mouse events are in map coordinate frame
        """
        self._last_mouse_pos = event.pos

    def _on_mouse_press(self, event):
        '''
        Vispy mouse button press callback.
        All mouse events are in map coordinate frame
        '''
        self._last_mouse_pos = event.pos

    def _on_mouse_release(self, event):
        """
        Vispy mouse button release callback.
        All mouse events are in map coordinate frame
        """

        self._last_mouse_pos = event.pos


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print "\n\n\nVispy 2-d template with images."

    s = VispyUI()

    vispy.app.run()
