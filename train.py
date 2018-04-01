import tensorflow as tf
import glob
import model_deploy
import os
from inceptionv2 import inception_resnet_v2_base, inception_resnet_v2_arg_scope

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
batch_size = 1
num_clones = 1
tf.logging.set_verbosity(tf.logging.INFO)

slim = tf.contrib.slim

is_training = True
np_branch1 = 26  # vec
np_branch2 = 14  # heat
size = 368
output_stride = 8
outsize = size // output_stride
stages = 6
weight_decay = 1e-9
lr = 1e-6


def conv(x, nf, ks, name, weight_decay):
    x = slim.batch_norm(x)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, nf, ks, name=name,
                         padding='same',
                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    return x


def stage1_block(x, num_p, branch, weight_decay):
    # Block 1
    x = conv(x, 128, 3, "Mconv1_stage1_L%d" % branch, weight_decay)
    x = conv(x, 128, 3, "Mconv2_stage1_L%d" % branch, weight_decay)
    x = conv(x, 128, 3, "Mconv3_stage1_L%d" % branch, weight_decay)
    x = conv(x, 512, 1, "Mconv4_stage1_L%d" % branch, weight_decay)
    x = conv(x, num_p, 1, "Mconv5_stage1_L%d" % branch, weight_decay)
    return x


def stageT_block(x, num_p, stage, branch, weight_decay):
    # Block 1
    x = conv(x, 128, 7, "Mconv1_stage%d_L%d" % (stage, branch), weight_decay)
    x = conv(x, 128, 7, "Mconv2_stage%d_L%d" % (stage, branch), weight_decay)
    x = conv(x, 128, 7, "Mconv3_stage%d_L%d" % (stage, branch), weight_decay)
    x = conv(x, 128, 7, "Mconv4_stage%d_L%d" % (stage, branch), weight_decay)
    x = conv(x, 128, 7, "Mconv5_stage%d_L%d" % (stage, branch), weight_decay)
    x = conv(x, 128, 1, "Mconv6_stage%d_L%d" % (stage, branch), weight_decay)
    x = conv(x, num_p, 1, "Mconv7_stage%d_L%d" % (stage, branch), weight_decay)
    return x




def _parse_function(example_proto):
    keys_to_features = {
        'mask_raw': tf.FixedLenFeature((), tf.string, default_value=''),
        'heat_raw': tf.FixedLenFeature((), tf.string, default_value=''),
        'vec_raw': tf.FixedLenFeature((), tf.string, default_value=''),
        'image_raw': tf.FixedLenFeature((), tf.string, default_value='')
    }
    parsed_features = tf.parse_single_example(
        example_proto, keys_to_features)
    img = parsed_features['image_raw']
    img = tf.decode_raw(img, tf.uint8)
    img.set_shape([size*size*3])
    img = tf.reshape(img, [size, size,3])
    mask = parsed_features['mask_raw']
    mask = tf.decode_raw(mask, tf.float32)
    mask.set_shape([outsize * outsize])
    mask = tf.reshape(mask, [outsize, outsize, 1])
    vec = parsed_features['vec_raw']
    vec = tf.decode_raw(vec, tf.float32)
    vec.set_shape([outsize * outsize * np_branch1])
    vec = tf.reshape(vec, [outsize, outsize, np_branch1])
    heat = parsed_features['heat_raw']
    heat = tf.decode_raw(heat, tf.float32)
    heat.set_shape([outsize * outsize * np_branch2])
    heat = tf.reshape(heat, [outsize, outsize, np_branch2])
    img = tf.cast(img, tf.float32)
    img /= 256
    img -= 0.5
    img *= 2
    # TODO:
    # generate data on the fly, add data rotation
    vec_mask = tf.concat([mask] * np_branch1, axis=-1)
    heat_mask = tf.concat([mask] * np_branch2, axis=-1)
    return img, vec_mask, heat_mask, vec, heat


def model_fn(inputs, vec_weight_input, heat_weight_input, vec_label, heat_label):
    # images are rescaled to -1~1 in resnet_v2
    with slim.arg_scope(inception_resnet_v2_arg_scope()):
        with tf.variable_scope('InceptionResnetV2', 'InceptionResnetV2', [inputs], reuse=None) as scope:
            with slim.arg_scope([slim.batch_norm], is_training=True):
                stage0_out, end_points = inception_resnet_v2_base(inputs, scope=scope)
    vec_weight_inputb = tf.equal(vec_weight_input, 1)
    heat_weight_inputb = tf.equal(heat_weight_input, 1)
    vec_label = tf.boolean_mask(vec_label, vec_weight_inputb)
    heat_label = tf.boolean_mask(heat_label, heat_weight_inputb)
    # feature map refine
    stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1, weight_decay)
    w1 = tf.boolean_mask(stage1_branch1_out, vec_weight_inputb)
    tf.losses.mean_squared_error(vec_label, w1)

    stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, weight_decay)
    w2 = tf.boolean_mask(stage1_branch2_out, heat_weight_inputb)
    tf.losses.mean_squared_error(heat_label, w2)

    net = tf.concat([stage1_branch1_out, stage1_branch2_out, stage0_out], -1)

    for sn in range(2, stages + 1):
        # stage SN - branch 1 (PAF)
        stageT_branch1_out = stageT_block(net, np_branch1, sn, 1, weight_decay)
        w1 = tf.boolean_mask(stageT_branch1_out, vec_weight_inputb)
        tf.losses.mean_squared_error(vec_label, w1)

        # stage SN - branch 2 (confidence maps)
        stageT_branch2_out = stageT_block(net, np_branch2, sn, 2, weight_decay)
        w2 = tf.boolean_mask(stageT_branch2_out, heat_weight_inputb)
        tf.losses.mean_squared_error(heat_label, w2)

        net = tf.concat(
            [stageT_branch1_out, stageT_branch2_out, stage0_out], -1)
    with tf.device(config.variables_device()):
        tf.summary.image('vec_pred', (stageT_branch1_out*vec_weight_input)[:, :, :, 0:1])
        tf.summary.image('heat_pred', (stageT_branch2_out*heat_weight_input)[:, :, :, 0:1])


if __name__ == '__main__':
    valdata = glob.glob('/media/xxx/Data/keypoint/data/records/val*')
    traindata = glob.glob('/media/xxx/Data/keypoint/data/records/train*')

    config = model_deploy.DeploymentConfig(
        num_clones=num_clones,
        clone_on_cpu=False
    )
    with tf.device(config.inputs_device()):
        dataset = tf.data.TFRecordDataset(traindata + valdata)
        # Parse the record into tensors.
        dataset = dataset.map(_parse_function, num_parallel_calls=4)
        dataset = dataset.shuffle(buffer_size=100)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()  # Repeat the input indefinitely.
        iterator = dataset.make_one_shot_iterator()

        inputs, vec_weight_input, heat_weight_input, vec_label, heat_label = iterator.get_next()
        tf.summary.image('input_image', inputs)
        tf.summary.image('mask', vec_weight_input[:, :, :, 0:1])
        tf.summary.image('vec', vec_label[:, :, :, 0:1])
        tf.summary.image('heat', heat_label[:, :, :, 0:1])

    with tf.device(config.variables_device()):
        global_step = tf.train.create_global_step()
        global_step = tf.cast(global_step, tf.int32)
    with tf.device(config.optimizer_device()):
        # lr = tf.train.piecewise_constant(
        #     global_step, [100000, 200000], [3e-4, 3e-5, 3e-6])
        tf.summary.scalar('learning_rate', lr)
        optimizer = tf.train.AdamOptimizer(lr)

    model_dp = model_deploy.deploy(config, model_fn, [
                                   inputs, vec_weight_input, heat_weight_input, vec_label, heat_label], optimizer=optimizer)

    variables_to_restore = slim.get_variables_to_restore(
        include=['.*InceptionResnetV2.*'], exclude=[".*Mixed_7a.*", '.*Adam.*', ".*Conv2d_7b_1x1.*", ".*Momentum.*"])
    init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
        './inception_resnet_v2_2016_08_30.ckpt', variables_to_restore)

    def init_fn(sess):
        sess.run(init_assign_op, init_feed_dict)

    slim.learning.train(
        model_dp.train_op,
        logdir='logs/{}'.format(1e-4),
        init_fn=init_fn,
        summary_op=model_dp.summary_op,
        save_interval_secs=300,
        save_summaries_secs=300,
        log_every_n_steps=10,
        session_config=tf.ConfigProto(allow_soft_placement=True)
    )
