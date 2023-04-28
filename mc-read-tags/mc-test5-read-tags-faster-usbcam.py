#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2020-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import sys

sys.path.append('../')
import gi
import configparser

gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
from ctypes import *
import time
import sys
import math
import platform
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
from common.FPS import PERF_DATA
import numpy as np
import pyds
import cv2
import os
import os.path
from os import path

# Text recognition imports/global variables
sys.path.append('text/')
from readTag import Options, create_digit_read_envir, webcam_img, batch_of_images

digit_model_path = "text/best_accuracy.pth"
reading_threshold = 0.5

print("Creating tag reading model")
opt = Options(digit_model_path)
converter, model, AlignCollate_demo = create_digit_read_envir(opt)


# Text to TrackerID datastructure
trackerID2Text = dict() # key: trackerID, value: tuple (bbox area * confidence, predicted text)
updateMultiplier = 1.1


# Deepstream global variables
perf_data = None
frame_count = {}
#saved_count = {}
global PGIE_CLASS_ID_NEAR
PGIE_CLASS_ID_NEAR = 0
global PGIE_CLASS_ID_DRINKING
PGIE_CLASS_ID_DRINKING = 2

MAX_DISPLAY_LEN = 64
PGIE_CLASS_ID_NEAR = 0
PGIE_CLASS_ID_FAR = 1
PGIE_CLASS_ID_DRINKING = 2
MUXER_OUTPUT_WIDTH = 1920
MUXER_OUTPUT_HEIGHT = 1080
MUXER_BATCH_TIMEOUT_USEC = 4000000
TILED_OUTPUT_WIDTH = 1920
TILED_OUTPUT_HEIGHT = 1080
GST_CAPS_FEATURES_NVMM = "memory:NVMM"
pgie_classes_str = ["Near", "Far", "Drinking"]



# tiler_sink_pad_buffer_probe  will extract metadata received on tiler src pad
# and update params for drawing rectangle, object information etc.
def tiler_sink_pad_buffer_probe(pad, info, u_data):
    frame_number = 0
    num_rects = 0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    l_frame = batch_meta.frame_meta_list

    # TAG INSERT
    framesToRead = dict() # maps frame number to frame of frames to be used for tag reading
    trackingIDsToUpdate = set() # set of trackingDs that will be updated during this batch
    toReadDict = dict() # maps trackingID to tuple of bbox coords/frame num
    objectMetasToUpdate = list() # list of object metas that will need updating after new tags have been read

    # END TAG INSERT


    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        l_obj = frame_meta.obj_meta_list
        num_rects = frame_meta.num_obj_meta
        obj_counter = {
            PGIE_CLASS_ID_NEAR: 0,
            PGIE_CLASS_ID_DRINKING: 0,
            PGIE_CLASS_ID_FAR: 0
        }

        shouldFrameBeRead = False

        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            obj_counter[obj_meta.class_id] += 1
            if obj_meta.class_id == PGIE_CLASS_ID_NEAR: # if near object
                
                if obj_meta.object_id not in trackerID2Text or obj_meta.rect_params.width * obj_meta.rect_params.height * obj_meta.confidence > trackerID2Text[obj_meta.object_id][0]*updateMultiplier: # if confidence*area is better than old confidence*area
                    shouldFrameBeRead = True
                    
                    # ensure the trackerID is marked as being edited
                    if obj_meta.object_id not in trackingIDsToUpdate:
                        trackingIDsToUpdate.add(obj_meta.object_id)
                    
                    # add/update trackerID's tag to be read
                    toReadDict[obj_meta.object_id] = (frame_number, obj_meta.rect_params.left, obj_meta.rect_params.top, obj_meta.rect_params.left + obj_meta.rect_params.width, obj_meta.rect_params.top + obj_meta.rect_params.height, obj_meta.object_id)
                    
                    # Update highest record in global variable, keep pred text
                    trackerID2Text[obj_meta.object_id] = (obj_meta.rect_params.width * obj_meta.rect_params.height * obj_meta.confidence, "0000")

                
                # object is near
                if obj_meta.object_id not in trackingIDsToUpdate:
                    try:
                        obj_meta.text_params.display_text = f"{trackerID2Text[obj_meta.object_id][1]}"
                    except KeyError as e:
                        print(f"KEY ERROR for {obj_meta.object_id}")
                else: # if trackingID is going to be updated
                    objectMetasToUpdate.append(obj_meta)

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        print("Frame Number=", frame_number, "Number of Objects=", num_rects, "Near_count=",
              obj_counter[PGIE_CLASS_ID_NEAR], "Drinking_count=", obj_counter[PGIE_CLASS_ID_DRINKING])


        # Save frame if needed
        if shouldFrameBeRead:
            n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            # convert python array into numpy array format in the copy mode.
            frame_copy = np.array(n_frame, copy=True, order='C')
            # convert the array into PIL usable format and save into frame dict
            framesToRead[frame_number] = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2RGB)

        # update frame rate through this probe
        stream_index = "stream{0}".format(frame_meta.pad_index)
        global perf_data
        perf_data.update_fps(stream_index)
        try:
            l_frame = l_frame.next
        except StopIteration:
            break


    # Do tag reading
    if len(toReadDict) > 0:
        toRead = list(toReadDict.values())
        frame_list, predicted_texts, predicted_confs = batch_of_images(opt, framesToRead, toRead, converter, model, AlignCollate_demo, doPrint=False)

        for i in range(len(predicted_texts)):
            # Update predicted text in the global data structure
            trackerID2Text[toRead[i][5]] = (trackerID2Text[toRead[i][5]][0], predicted_texts[i])

        for cur_obj_meta in objectMetasToUpdate:
            # Update texts displayed to screen with new texts
            cur_obj_meta.text_params.display_text = trackerID2Text[cur_obj_meta.object_id][1]


    return Gst.PadProbeReturn.OK



def main(args):
    # Check input arguments
    if len(args) != 2:
        sys.stderr.write("usage: %s <v4l2-device-path> \n" % args[0])
        sys.exit(1)

    global perf_data
    perf_data = PERF_DATA(len(args) - 1)
    number_sources = len(args) - 1

    # Standard GStreamer initialization
    Gst.init(None)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    print("Creating Source\n")
    source = Gst.ElementFactory.make("v4l2src", "usb-cam-source")
    if not source:
        sys.stderr.write("Unable to create source \n")

    caps_v4l2src = Gst.ElementFactory.make("capsfilter", "v4l2src_caps")
    if not caps_v4l2src:
        sys.stderr.write(" Unable to create v4l2src capsfilter \n")


    print("Creating Video Converter \n")

    # Adding videoconvert -> nvvideoconvert as not all
    # raw formats are supported by nvvideoconvert;
    # Say YUYV is unsupported - which is the common
    # raw format for many logi usb cams
    # In case we have a camera with raw format supported in
    # nvvideoconvert, GStreamer plugins' capability negotiation
    # shall be intelligent enough to reduce compute by
    # videoconvert doing passthrough (TODO we need to confirm this)


    # videoconvert to make sure a superset of raw formats are supported
    vidconvsrc = Gst.ElementFactory.make("videoconvert", "convertor_src1")
    if not vidconvsrc:
        sys.stderr.write(" Unable to create videoconvert \n")

    # nvvideoconvert to convert incoming raw buffers to NVMM Mem (NvBufSurface API)
    nvvidconvsrc = Gst.ElementFactory.make("nvvideoconvert", "convertor_src2")
    if not nvvidconvsrc:
        sys.stderr.write(" Unable to create Nvvideoconvert \n")

    caps_vidconvsrc = Gst.ElementFactory.make("capsfilter", "nvmm_caps")
    if not caps_vidconvsrc:
        sys.stderr.write(" Unable to create capsfilter \n")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")


    print("Creating Pgie \n ")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")
    
    print("Creating tracker \n")
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not tracker:
        sys.stderr.write(" Unable to create tracker \n")



    print("Creating nvvidconv \n ")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")
    
    print("Creating nvosd \n ")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")
    
    if (is_aarch64()):
        print("Creating transform \n ")
        transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")
        if not transform:
            sys.stderr.write(" Unable to create transform \n")

    print("Creating EGLSink \n")
    sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    if not sink:
        sys.stderr.write(" Unable to create egl sink \n")

    print("Playing cam %s " %args[1])
    caps_v4l2src.set_property('caps', Gst.Caps.from_string("video/x-raw, framerate=30/1"))
    caps_vidconvsrc.set_property('caps', Gst.Caps.from_string("video/x-raw(memory:NVMM)"))
    source.set_property('device', args[1])
    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 4000000)
    pgie.set_property('config-file-path', "config_infer_primary_yoloV5.txt")
    
    #Set properties of tracker
    config = configparser.ConfigParser()
    config.read('dstest2_tracker_config.txt')
    config.sections()

    for key in config['tracker']:
        if key == 'tracker-width' :
            tracker_width = config.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        if key == 'tracker-height' :
            tracker_height = config.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        if key == 'gpu-id' :
            tracker_gpu_id = config.getint('tracker', key)
            tracker.set_property('gpu_id', tracker_gpu_id)
        if key == 'll-lib-file' :
            tracker_ll_lib_file = config.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        if key == 'll-config-file' :
            tracker_ll_config_file = config.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)
        if key == 'enable-batch-process' :
            tracker_enable_batch_process = config.getint('tracker', key)
            tracker.set_property('enable_batch_process', tracker_enable_batch_process)
        if key == 'enable-past-frame' :
            tracker_enable_past_frame = config.getint('tracker', key)
            tracker.set_property('enable_past_frame', tracker_enable_past_frame)




    sink.set_property("sync", False)

    print("Adding elements to Pipeline \n")
    pipeline.add(source)
    pipeline.add(caps_v4l2src)
    pipeline.add(vidconvsrc)
    pipeline.add(nvvidconvsrc)
    pipeline.add(caps_vidconvsrc)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    if is_aarch64():
        pipeline.add(transform)
    pipeline.add(sink)

    print("Linking elements in the Pipeline \n")
    source.link(caps_v4l2src)
    caps_v4l2src.link(vidconvsrc)
    vidconvsrc.link(nvvidconvsrc)
    nvvidconvsrc.link(caps_vidconvsrc)

    sinkpad = streammux.get_request_pad("sink_0")
    if not sinkpad:
        sys.stderr.write(" Unable to get the sink pad of streammux \n")
    srcpad = caps_vidconvsrc.get_static_pad("src")
    if not srcpad:
        sys.stderr.write(" Unable to get source pad of caps_vidconvsrc \n")
    srcpad.link(sinkpad)
    streammux.link(pgie)

    pgie.link(tracker)
    tracker.link(nvvidconv)
    nvvidconv.link(nvosd)
    if is_aarch64():
        nvosd.link(transform)
        transform.link(sink)
    else:
        nvosd.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    osd_sink_pad = nvosd.get_static_pad("sink")
    if not osd_sink_pad:
        sys.stderr.write(" Unable to get src pad \n")
    else:
        osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, tiler_sink_pad_buffer_probe, 0)
        # perf callback function to print fps every 5 sec
        GLib.timeout_add(5000, perf_data.perf_print_callback)


    print("Starting pipeline \n")
    # start play back and listed to events		
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
