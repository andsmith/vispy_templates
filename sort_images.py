import cv2
import multiprocessing
import json
import time
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


def shrink_image(img, max_dim):
    if img.shape[0] > max_dim:
        new_height = max_dim
        new_width = int((new_height / float(img.shape[0])) * img.shape[1])
    else:
        new_height, new_width = img.shape[:2]
    if new_width > max_dim:
        newer_width = max_dim
        newer_height = int((newer_width / float(new_width)) * new_height)
    else:
        newer_height, newer_width = new_height, new_width
    if newer_width != img.shape[1] or newer_height != img.shape[0]:
        nimg = cv2.cv2.resize(img, (newer_width, newer_height))
        img = nimg
    return img

'''
class ImageMemCache(object):
    """
    Load images in separate thread, Nearest Neighbors strategy
    """
    IMAGE_EXTENSIONS = ['.jpg', '.png']
    MAX_IMG_DIM = 8191

    def __init__(self, in_dir, max_size=0, start_index=0):

        self._dir = in_dir
        self._image_files = [f for f in os.listdir(self._dir) if os.path.splitext(f)[1] in self.IMAGE_EXTENSIONS]

        self._index = start_index
        if self._index > len(self._image_files):
            logging.warn("Starting index cannot be > number of files found (%s), setting to 0." % (len(self._image_files),))
            self._index = 0
        self._cache = {}
        self._max_size = max_size
        self._update_cache()

    def _update_cache(self):
        """
        Make sure closest n items to current index are loaded:
        """
        closest = np.argsort(np.abs(np.arange(0, len(self._image_files)) - self._index))
        if self._max_size > 0:
            closest = closest[:self._max_size]
        cached = self._cache.keys()
        evictees = [c for c in cached if c not in closest]

        for close in closest:
            if close not in self._cache:
                logging.info("Image cache - loading:  %s" % (close,))
                self._cache[close] = self._read_item_n(close)

        for e in evictees:
            logging.info("Image cache - evicting:  %s" % (e,))
            del self._cache[e]

    def _read_item_n(self, n):
        image_fullpath = os.path.join(self._dir, self._image_files[n])
        img = cv2.imread(image_fullpath)[:, :, ::-1]
        return shrink_image(img, self.MAX_IMG_DIM)

    def get_item_n(self, n):
        """
        Return image and filename of the n'th item
        :return: (image, filename) 
        """
        self._index = n
        self._update_cache()
        return self._cache[n], self._image_files[n]

    def get_n(self):
        return len(self._image_files)

    def get_item_f(self, f):
        """
        Return image and index of the item with filename f
        :return: (image, index) 
        """
        inds = [i for i, fn in enumerate(self._image_files) if fn == f]
        if len(inds) == 0:
            raise Exception("No image file with name %s found in input path %s" % (f, self._dir))
        return self.get_item_n(inds[0])[0], inds[0]
'''




class ImageSorter(object):
    """
    Display an image, rectangle, text, enable pan/tilt, keyboard
    """
    def __init__(self, image_dir, out_file=None, max_load=10):
        self._max_load = max_load
        self._in_dir = os.path.abspath(os.path.expanduser(image_dir))
        self._out_file = out_file if out_file is not None else os.path.join(self._in_dir, "results.json")
        self._load_results()
        self._images = ImageMemCache(self._in_dir, max_size=max_load)

        # state
        self._last_mouse_pos = np.array([0.0, 0.0])
        self._canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, title="Route editor -- H for help")
        self._viewbox = self._canvas.central_widget.add_view()
        self._viewbox.camera = vispy.scene.cameras.PanZoomCamera(parent=self._viewbox.scene, aspect=1)
        self._add_graphics_elements()
        self._canvas.events.key_press.connect(self._on_key_press)
        self._canvas.events.mouse_move.connect(self._on_mouse_move)
        self._canvas.events.mouse_press.connect(self._on_mouse_press)
        self._canvas.events.mouse_release.connect(self._on_mouse_release)
        self._canvas.events.resize.connect(self._on_resize)
        #import ipdb; ipdb.set_trace()
        self._update_image()

    def _load_results(self):
        self._current_index = 0

        if os.path.exists(self._out_file):
            with open(self._out_file, 'r') as infile:
                data = json.load(infile)
            self._results = data['results']
        else:
            self._results = []
            self._unlabeled = 

    def _save_results(self):
        with open(self._out_file, 'w') as outfile:
            json.dump({'current_index': self._current_index,
                       'results': self._results}, outfile)

    def _add_graphics_elements(self):
        """
        Create all the graphics objects (VISPY objects), put them on the canvas.
        """

        # Image
        self._image_object = Image(None, parent=self._viewbox.scene)
        self._image_object.set_gl_state('translucent', depth_test=False)
        self._image_object.order = 1
        self._image_object.visible = True

        self._text_box_width = 150
        self._text_box_height = 60
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
        self._resize_text_bkg_box()

        # Text
        self._font1_size = 10
        self._font2_size = 18
        self._vspace = self._font1_size*2.2
        self._text_pos = [self._text_box_offset + 10, self._text_box_offset+10]
        self._text_obj = vispy.scene.visuals.Text("",
                                                     parent=self._canvas.scene,
                                                     color=[0.9, 0.8, 0.8],
                                                     anchor_x='left',
                                                     anchor_y='top')
        self._text_obj.pos = self._text_pos
        self._text_obj.font_size = self._font1_size
        self._text_obj.visible = True
        self._text_obj.order = 3

        self._text2_pos = [self._text_pos[0], self._vspace + self._text_pos[1]]
        self._text2_obj = vispy.scene.visuals.Text("",
                                                     parent=self._canvas.scene,
                                                     color=[0.9, 0.8, 0.8],
                                                     anchor_x='left',
                                                     anchor_y='top')
        self._text2_obj.pos= self._text2_pos
        self._text2_obj.font_size = self._font2_size
        self._text2_obj.visible = True
        self._text2_obj.order = 3

    def _resize_text_bkg_box(self):
        logging.info('resize')
        self._text_bkg_rect.center = np.array([self._canvas.size[0]/2, self._text_box_height / 2+self._text_box_offset])
        self._text_bkg_rect.width = self._canvas.size[0] - 2 * self._text_box_offset

    def _on_resize(self, event):
        self._resize_text_bkg_box()


    def _update_image(self):
        """
        Extract an image to display from the current cost map.
        """

        self._current_image, self._current_filename = self._images.get_item_n(self._current_index)
        logging.info("Got new image:  %s, %s" % (self._current_image.shape, self._current_filename))
        xmin, ymin = 0, 0
        xmax, ymax = self._current_image.shape[1], self._current_image.shape[0]
        self._viewbox.camera.set_range(x=(xmin, xmax), y=(ymin, ymax), z=None)
        self._image_object.set_data(self._current_image)
        self._image_object.visible = True
        self._update_text()

    def _update_text(self):
        unique_results = get_latest_results(self._results)
        result = unique_results[self._current_index] if self._current_index in unique_results else "(unlabeled)"
        text1 = self._current_filename
        text2 = "%i / %i - label:  %s" % (self._current_index, self._images.get_n(), result)
        self._text_obj.text = text1
        self._text2_obj.text = text2

        self._text_obj.visible = True
        self._text2_obj.visible = True

    def _on_key_press(self, ev):
        """
        vispy keyboard callback
        """

        if not ev or not ev.key:
            # on Mac sometimes this callback is triggered with 
            # NoneType key when clicking between desktops or full screen apps
            return

        # control_pressed = 'Control' in [e.name for e in ev.modifiers]
        # shift_pressed = 'Shift' in [e.name for e in ev.modifiers]
        # print "Shift:  %s\nControl:%s" % (shift_pressed, control_pressed)

        if ev.key.name == "Left":
            self._advance_index(-1)

        elif ev.key.name == "Right":
            self._advance_index(1)

        elif ev.key.name == "X":
            self._classify_and_advance(0)

        elif ev.key.name == "O":
            self._classify_and_advance(1)

        else:
            logging.info("Unknown keypress:  %s" % (ev.key.name, ))

    def _advance_index(self, direction):
        if self._current_index + direction >= 0 and self._current_index + direction <= self._images.get_n() - 1:
            self._current_index = self._current_index + direction
        self._update_image()
        self._update_text()

    def _classify_and_advance(self, label):
        self._results.append({'index': self._current_index,
                              'file': self._current_filename,
                              'label': label,
                              'time': time.time()})
        self._save_results()
        self._advance_index(1)


    def _on_mouse_move(self, event):
        self._last_mouse_pos = event.pos

    def _on_mouse_press(self, event):
        self._last_mouse_pos = event.pos

    def _on_mouse_release(self, event):
        self._last_mouse_pos = event.pos

def get_latest_results(results):
    labels = {}
    for r in results:
        labels[r['index']] = r['label']
    return labels

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Hand-sort/categorize image set", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("image_dir", help="source location of images", type=str)
    parser.add_argument("-o", "--output", help="Results file, default is image_dir/sorted.json", type=str)
    parser.add_argument("-l", "--load", help="Pre-Load at most this many images.", type=int, default=0)
    parsed = parser.parse_args()

    s = ImageSorter(image_dir = parsed.image_dir, out_file = parsed.output, max_load= parsed.load)
    vispy.app.run()
