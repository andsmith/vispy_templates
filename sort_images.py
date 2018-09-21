import time
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

IMAGE_EXTENSIONS = ['.png', '.jpg']


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


def _get_latest_labeling(label_list):
    """
    :param label_list:  list of dict {'file': filename, 'label': int, 'timestamp': float (epoch)} 
    :return: dict{'file': 
    """
    labels = {}
    sorted_lables = sorted(label_list, key=lambda x: x['timestamp'])
    for l in sorted_lables:
        labels[l['file']] = l
    return labels

class LabeledImageSet(object):
    """
    Mange image filenames, labels, results file
    """
    def __init__(self, image_dir, in_file, out_file, cache=10):
        self._in_file = in_file
        self._out_file = out_file
        self._image_cache = []
        self._cache_size=cache
        # read labels
        if os.path.exists(in_file):
            with open(in_file, 'r') as infile:
                self._existing_labels = json.load(infile)
        else:
            self._existing_labels = []
        self._new_labels = []
        current_labels = _get_latest_labeling(self._existing_labels)

        self._image_dir = image_dir
        extensions = IMAGE_EXTENSIONS + [e.upper() for e in IMAGE_EXTENSIONS]
        self._image_files = [f for f in os.listdir(self._image_dir) if os.path.splitext(f)[1] in extensions]

        self._labeled_files = [f for f in current_labels]
        self._unlabeled_files = [f for f in self._image_files if f not in current_labels]

        logging.info("Read %i image files." % (len(self._image_files),))
        logging.info("Images labeled:  %i" % (len(self._existing_labels),))
        logging.info("Images remaining:  %i" % (len(self._unlabeled_files),))

    def save(self):
        labels = self._existing_labels + self._new_labels
        logging.info("Writing %i + %i = %i labels." % (len(self._existing_labels), len(self._new_labels), len(labels)))
        with open(out_file,'w') as outfile:
            json.dump(labels, outfile)

    def get_unlabeled(self):
        if len(self._unlabeled_files) == 0:
            return self._image_files[0], 0

        return self._unlabeled_files[0], self.get_index(self._unlabeled_files[0])

    def get_index(self, filename):
        ind = [i for i, f in enumerate(self._image_files) if f == filename]
        return ind[0]

    def get_i(self, i):
        return self._image_files[i]

    def get_image(self, filename):
        indices = [cache_pair for cache_pair in self._image_cache if cache_pair['filename'] == filename]
        if len(indices) == 1:
            return indices[0]['image']
        img = shrink_image(cv2.imread(os.path.join(self._image_dir, filename)), 8191)[:,:,::-1]

        if len(self._image_cache) >= self._cache_size:
            self._image_cache.pop(0)
        self._image_cache.append({'image': img, 'filename': filename})
        return img

    def get_n(self):
        return len(self._image_files)

    def apply_label(self, filename, label):
        if filename in self._unlabeled_files:
            #print "Removed from unlabeled:", filename
            #print len(self._unlabeled_files), filename in self._unlabeled_files
            self._unlabeled_files.remove(filename)
            #print len(self._unlabeled_files), filename in self._unlabeled_files

        if filename not in self._labeled_files:
            #print "Added to labeled:", filename
            self._labeled_files.append(filename)
        self._new_labels.append({'file': filename,
                                 'label': label,
                                 'timestamp': time.time()})

    def get_label(self, filename):
        if filename not in self._labeled_files:
            return None
        labels = _get_latest_labeling(self._existing_labels + self._new_labels)
        return labels[filename]['label']


class ImageSorter(object):
    """
    Display an image, rectangle, text, enable pan/tilt, keyboard
    """

    def __init__(self, image_dir, out_file=None, max_load=10):
        self._in_dir = os.path.abspath(os.path.expanduser(image_dir))
        self._out_file = out_file if out_file is not None else os.path.join(self._in_dir, "results.json")
        self._images = LabeledImageSet(image_dir, self._out_file, self._out_file)
        self._current_filename, self._current_index = self._images.get_unlabeled()

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
        self._update_image()

    def _save_results(self):
        self._images.save()

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
        self._text_bkg_rect = vispy.scene.visuals.Rectangle([self._text_box_width / 2 + self._text_box_offset,
                                                             self._text_box_height / 2 + self._text_box_offset],
                                                            color=[0.1, 0.0, 0.0, .8],
                                                            border_color=[0.1, 0.0, 0.0],
                                                            border_width=2,
                                                            height=self._text_box_height,
                                                            width=self._text_box_width,
                                                            radius=10.0,
                                                            parent=self._canvas.scene)
        self._text_bkg_rect.set_gl_state('translucent', depth_test=False)
        self._text_bkg_rect.visible = True
        self._text_bkg_rect.order = 2
        self._resize_text_bkg_box()

        # Text
        self._font1_size = 10
        self._font2_size = 18
        self._vspace = self._font1_size * 2.2
        self._text_pos = [self._text_box_offset + 10, self._text_box_offset + 10]
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
        self._text2_obj.pos = self._text2_pos
        self._text2_obj.font_size = self._font2_size
        self._text2_obj.visible = True
        self._text2_obj.order = 3

    def _resize_text_bkg_box(self):
        logging.info('resize')
        self._text_bkg_rect.center = np.array(
            [self._canvas.size[0] / 2, self._text_box_height / 2 + self._text_box_offset])
        self._text_bkg_rect.width = self._canvas.size[0] - 2 * self._text_box_offset

    def _on_resize(self, event):
        self._resize_text_bkg_box()

    def _update_image(self):
        """
        Extract an image to display from the current cost map.
        """
        self._current_filename = self._images.get_i(self._current_index)
        self._current_image = self._images.get_image(self._current_filename)

        logging.info("Got new image:  %s, %s" % (self._current_image.shape, self._current_filename))
        xmin, ymin = 0, 0
        xmax, ymax = self._current_image.shape[1], self._current_image.shape[0]
        self._viewbox.camera.set_range(x=(xmin, xmax), y=(ymin, ymax), z=None)
        self._image_object.set_data(self._current_image)
        self._image_object.visible = True
        self._update_text()

    def _update_text(self):
        result = self._images.get_label(self._current_filename)
        result_str = "%i" % (result,) if result is not None else "(unlabeled)"

        text1 = self._current_filename
        text2 = "%i / %i - label:  %s" % (self._current_index, self._images.get_n(), result_str)
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
        shift_pressed = 'Shift' in [e.name for e in ev.modifiers]
        # print "Shift:  %s\nControl:%s" % (shift_pressed, control_pressed)

        if ev.key.name == "Left":
            self._advance_index(-1)

        elif ev.key.name == "Right":
            self._advance_index(1)

        elif ev.key.name == "X":
            self._classify_and_go_to_next(0, skip_to_next_unlabeled=not shift_pressed)

        elif ev.key.name == "O":
            self._classify_and_go_to_next(1, skip_to_next_unlabeled=not shift_pressed)

        else:
            logging.info("Unknown keypress:  %s" % (ev.key.name,))

    def _advance_index(self, direction):
        new_index = self._current_index + direction
        if new_index >= 0 and new_index <= self._images.get_n() - 1:
            self._current_index = new_index
        self._update_image()
        self._update_text()

    def _classify_and_go_to_next(self, label, skip_to_next_unlabeled=True):
        self._images.apply_label(self._current_filename, label)
        self._images.save()
        if skip_to_next_unlabeled:
            self._current_filename, self._current_index = self._images.get_unlabeled()
            self._update_image()
            self._update_text()
        else:
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
    parser.add_argument("-o", "--output", help="Results file, default is image_dir/../images_sorted.json", type=str)
    parser.add_argument("-l", "--load", help="Pre-Load at most this many images.", type=int, default=0)
    parsed = parser.parse_args()
    image_dir = os.path.abspath(os.path.expanduser(parsed.image_dir))
    out_file = os.path.join(os.path.split(image_dir)[0], 'images_sorted.json')
    logging.info("Input:  %s" % (image_dir,))
    logging.info("Output:  %s" % (out_file,))

    s = ImageSorter(image_dir = image_dir, out_file = out_file, max_load= parsed.load)
    vispy.app.run()
