# imports
from styx_msgs.msg import TrafficLight
import os
# from object_detection.utils import label_map_util
# from object_detection.utils import visualization_utils as vis_util
import tensorflow as tf
import numpy as np
import time

class TLClassifier(object):
    def __init__(self, simulation):

        # default status
        self.current_status = TrafficLight.UNKNOWN

        #
        working_dir = os.path.dirname(os.path.realpath(__file__))

        # flag to switch between real and sim trained classifier
        self.simulation = simulation

        # Load the right models
        if self.simulation is True:
            self.checkpoint = working_dir + '/output_inference_graph_bosch_2_sim.pb/frozen_inference_graph.pb'
        else:
            self.checkpoint = working_dir + '/output_inference_graph_bosch_2_udacity_real.pb/frozen_inference_graph.pb'

        # Create a label dictionary
        item_green = {'id': 1, 'name': u'traffic_light-green'}
        item_red = {'id': 2, 'name': u'traffic_light-red'}
        item_yellow = {'id': 3, 'name': u'traffic_light-yellow'}

        self.label_dict = {1: item_green, 2: item_red, 3: item_yellow}

        # Build the model
        self.image_np_output = None
        self.detection_graph = tf.Graph()

        self.current_light = TrafficLight.UNKNOWN

        # create config
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # Create graph
        with self.detection_graph.as_default():
            graph_definition = tf.GraphDef()

            # load pre trained model
            with tf.gfile.GFile(self.checkpoint, 'rb') as fid:
                serial_graph = fid.read()
                graph_definition.ParseFromString(serial_graph)
                tf.import_graph_def(graph_definition, name='')

            # Create a reusable sesion attribute
            self.sess = tf.Session(graph=self.detection_graph, config=config)

        # get parameters by names
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # scores
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        self.activated = True  # flag to turn off classifier during development

        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        # Run the classifier if activated flag is true
        if self.activated is True:

            # create image as np.ndarray
            np_exp_image = np.expand_dims(image, axis=0)

            # get the detections and scores and bounding boxes
            with self.detection_graph.as_default():
                (boxes, scores, classes, num) = self.sess.run( [self.detection_boxes, self.detection_scores,
                                                                self.detection_classes, self.num_detections],
                                                               feed_dict={self.image_tensor: np_exp_image})

            # create np arrays
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes).astype(np.int32)

            # Set a Classification threshold
            classification_threshold = .50

            # Iterate the boxes to get all detections
            for i in range(boxes.shape[0]):

                # Get class name for detections with high enough scores
                if scores is None or scores[i] > classification_threshold:
                    class_name = self.label_dict[classes[i]]['name']

                    # Set default state to unknown
                    self.current_light = TrafficLight.UNKNOWN

                    if class_name == 'traffic_light-red':
                        self.current_light = TrafficLight.RED
                    elif class_name == 'traffic_light-green':
                        self.current_light = TrafficLight.GREEN
                    elif class_name == 'traffic_light-yellow':
                        self.current_light = TrafficLight.YELLOW

                    # Depth estimation
                    # Disabled because /vehicle/traffic_lights topic is available to waypoint updater
                    # Detected light is assumed to be the closest one


            # Visualization of the results of a detection
            # Disabled to remove dependency on aobject detection API
            # vis_util.visualize_boxes_and_labels_on_image_array(image, boxes, classes, scores,
            #                                                    self.label_dict,
            #                                                    use_normalized_coordinates=True,
            #                                                    line_thickness=8)

        # Set it to object attribute for visualization topic output
        # Can be disabled to gain a few ms in performance
        self.image_np_output = image

        return self.current_light
