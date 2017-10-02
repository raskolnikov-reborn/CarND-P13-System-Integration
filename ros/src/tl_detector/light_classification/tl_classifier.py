# imports
from styx_msgs.msg import TrafficLight
import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
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

        # Load labels and classification parameters
        self.path_to_labels = working_dir + '/tl_label_map.pbtxt'
        self.num_classes = 3

        self.label_map = label_map_util.load_labelmap(self.path_to_labels)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.num_classes,
                                                                    use_display_name=True)

        self.category_index = label_map_util.create_category_index(self.categories)

        # Build the model
        self.image_np_deep = None
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

        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        run_network = True  # flag to disable running network if desired

        if run_network is True:
            np_expanded_image = np.expand_dims(image, axis=0)

            # time0 = time.time()
            # Run Detection
            with self.detection_graph.as_default():
                (boxes, scores, classes, num) = self.sess.run(
                    [self.detection_boxes, self.detection_scores,
                     self.detection_classes, self.num_detections],
                    feed_dict={self.image_tensor: np_expanded_image})

            # time1 = time.time()
            #
            # print("Time in milliseconds", (time1 - time0) * 1000)


            # squeeze as numpy arrays
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes).astype(np.int32)

            min_score_thresh = .50


            for i in range(boxes.shape[0]):
                if scores is None or scores[i] > min_score_thresh:
                    class_name = self.category_index[classes[i]]['name']

                    # print('{}'.format(class_name))

                    # Traffic light thing
                    self.current_light = TrafficLight.UNKNOWN

                    if class_name == 'traffic_light-red':
                        self.current_light = TrafficLight.RED
                    elif class_name == 'traffic_light-green':
                        self.current_light = TrafficLight.GREEN
                    elif class_name == 'traffic_light-yellow':
                        self.current_light = TrafficLight.YELLOW

                    fx = 1345.200806
                    fy = 1353.838257
                    perceived_width_x = (boxes[i][3] - boxes[i][1]) * 800
                    perceived_width_y = (boxes[i][2] - boxes[i][0]) * 600

                    # ymin, xmin, ymax, xmax = box
                    # depth_prime = (width_real * focal) / perceived_width
                    # traffic light is 4 feet long and 1 foot wide?
                    perceived_depth_x = ((1 * fx) / perceived_width_x)
                    perceived_depth_y = ((3 * fy) / perceived_width_y)

                    estimated_distance = round((perceived_depth_x + perceived_depth_y) / 2)


                    # print("perceived_width", perceived_width_x, perceived_width_y)
                    # print("perceived_depth", perceived_depth_x, perceived_depth_y)
                    # print("Average depth (ft?)", estimated_distance)

            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image, boxes, classes, scores,
                self.category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

        # For visualization topic output
        self.image_np_deep = image

        return self.current_light
