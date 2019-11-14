import os, sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../spinup"))\

from spinup.utils.logx import restore_tf_graph
import tensorflow as tf

if __name__ == '__main__':
	sess = tf.Session()

	print(restore_tf_graph(sess, os.path.join(dir_path, "vpg_output_dir/simple_save")))
