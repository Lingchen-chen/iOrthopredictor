import tensorflow as tf
import numpy as np
from extern.vgg.vgg import vgg_19

slim = tf.contrib.slim


def content_loss(gt_end_points, renderer_end_points, weights):

    total_content_loss = np.float32(0.0)
    content_loss_dict = {}

    for name, weight in weights.items():
        # Reducing over all but the batch axis before multiplying with the weight
        try:
            loss = tf.reduce_sum(tf.abs(gt_end_points[name] - renderer_end_points[name]), [1, 2, 3])
            weighted_loss = tf.reduce_mean(weight * loss)
            loss = tf.reduce_mean(loss)

            content_loss_dict['content_loss/' + name] = loss
            content_loss_dict['weighted_content_loss/' + name] = weighted_loss
            total_content_loss += weighted_loss
        except:
            continue

    content_loss_dict['total_content_loss'] = total_content_loss

    return total_content_loss, content_loss_dict


def reconstruct_loss(x_aug, r_all, content_weights):

    gt_endpoints = vgg_19(x_aug, pooling="avg", final_endpoint="relu3_4")
    re_endpoints = vgg_19(r_all, pooling="avg", final_endpoint="relu3_4")
    total_content_loss, _ = content_loss(gt_endpoints, re_endpoints, content_weights)
    l1_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(r_all - x_aug), axis=[1, 2, 3]) * content_weights['relu_0'])
    # l1_loss = 0.
    return total_content_loss, l1_loss


def adversarial_loss(D, real, fake, real_msk):
    scores_real = D(real)
    scores_fake = D(fake)

    def D_logistic_r1(real_scores_out, fake_scores_out, gamma=10.0):
        loss = tf.nn.softplus(fake_scores_out)
        loss += tf.nn.softplus(-real_scores_out)

        with tf.name_scope('GradientPenalty'):
            real_grads = tf.gradients(tf.reduce_sum(real_scores_out), [real])[0]
            gradient_penalty = tf.reduce_sum(tf.square(real_grads * real_msk), axis=[1, 2, 3])
            reg = gradient_penalty * (gamma * 0.5)
        return tf.reduce_mean(loss), tf.reduce_mean(reg)

    def G_logistic_ns(fake_scores_out):
        loss = tf.nn.softplus(-fake_scores_out)  # -log(sigmoid(fake_scores_out))
        return tf.reduce_mean(loss), None

    D_loss, D_reg = D_logistic_r1(scores_real, scores_fake)
    G_loss, G_reg = G_logistic_ns(scores_fake)

    return D_loss, D_reg, G_loss, G_reg


def latent_kl_loss(q, p):
    mean1 = q
    mean2 = p

    kl = 0.5 * tf.square(mean2 - mean1)
    kl = tf.reduce_sum(kl, axis=[1,2,3])
    kl = tf.reduce_mean(kl)
    return kl


def normal_kl_loss(q):
    mean = q
    kl = 0.5 * tf.square(mean - 0)
    kl = tf.reduce_mean(kl)
    return kl