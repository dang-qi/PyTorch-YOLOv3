import tensorflow as tf
from tensorboardX import SummaryWriter


class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        #self.writer = tf.summary.FileWriter(log_dir)
        #self.writer = tf.summary.create_file_writer(log_dir)
        self.writer = SummaryWriter(logdir=log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        #summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        #self.writer.add_summary(summary, step)
        #with self.writer.as_default():
        #    tf.summary.scalar(tag, value, step=step)
        #    self.writer.flush()
        self.writer.add_scalar(tag, value, global_step=step)

    def list_of_scalars_summary(self, tag_value_pairs, step):
        """Log scalar variables."""
        #summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value) for tag, value in tag_value_pairs])
        #self.writer.add_summary(summary, step)
        #self.writer.flush()
        #with self.writer.as_default():
        #    for tag, value in tag_value_pairs:
        #        tf.summary.scalar(tag, value, step=step)
        #    self.writer.flush()
        for tag, value in tag_value_pairs:
            self.writer.add_scalar(tag, value, global_step=step)

    def get_network_graph(self, model, input_im=None):
        self.writer.add_graph(model, input_to_model=input_im)