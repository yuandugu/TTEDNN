from keras import backend as K
import tensorflow as tf
from tensorflow.python.ops import array_ops


def variant_focal_loss(gamma=2., alpha=0.25, rescale=False):
    """
    version 1

    """
    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """
        Focal loss for bianry-classification
        FL(p_t)=-rescaled_factor*alpha_t*(1-p_t)^{gamma}log(p_t)

        Notice:
        y_pred is probability after sigmoid

        Arguments:
            y_true {tensor} -- groud truth label, shape of [batch_size, 1]
            y_pred {tensor} -- predicted label, shape of [batch_size, 1]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {0.5})

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)
        model_out = tf.clip_by_value(y_pred, epsilon, 1.-epsilon)
        
        # compute cross entropy ce = ce_0 + ce_1 = - (1-y)*log(1-y_hat) - y*log(y_hat)
        ce_0 = tf.multiply(tf.subtract(1., y_true), -tf.math.log(tf.subtract(1., model_out)))
        ce_1 = tf.multiply(y_true, -tf.math.log(model_out))

        # compute focal loss fl = fl_0 + fl_1
        fl_0 = tf.multiply(tf.pow(model_out, gamma), ce_0)
        fl_0 = tf.multiply(1.-alpha, fl_0)
        fl_1 = tf.multiply(tf.pow(tf.subtract(1., model_out), gamma), ce_1)
        fl_1 = tf.multiply(alpha, fl_1)
        fl = tf.add(fl_0, fl_1)
        f1_avg = tf.reduce_mean(fl)
        
        if rescale:
            # rescale f1 to keep the quantity as ce
            ce = tf.add(ce_0, ce_1)
            ce_avg = tf.reduce_mean(ce)
            rescaled_factor = tf.divide(ce_avg, f1_avg + epsilon)
            f1_avg = tf.multiply(rescaled_factor, f1_avg)
        
        return f1_avg
    
    return focal_loss_fixed


def class_balanced_sigmoid_cross_entropy(logits, label, name='cross_entropy_loss'):
    """
    The class-balanced cross entropy loss,

    as in `Holistically-Nested Edge Detection

    <http://arxiv.org/abs/1504.06375>`_.

    This is more numerically stable than class_balanced_cross_entropy


    :param logits: size: the logits.

    :param label: size: the ground truth in {0,1}, of the same shape as logits.

    :returns: a scalar. class-balanced cross entropy loss
    """
    y = tf.cast(label, tf.float32)

    count_neg = tf.reduce_sum(1. - y) # the number of 0 in y
    count_pos = tf.reduce_sum(y) # the number of 1 in y (less than count_neg)
    beta = count_neg / (count_neg + count_pos)
 
    pos_weight = beta / (1 - beta)
    cost = tf.nn.weighted_cross_entropy_with_logits(logits, y, pos_weight)
    cost = tf.reduce_mean(cost * (1 - beta), name=name)
 
    return cost


def focal_loss_2(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
    
    """Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
    
    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)
    
    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent)


def focal_loss_3(gamma=2., alpha=.25):

    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))

    return focal_loss_fixed
