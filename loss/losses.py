# -*-coding:utf-8-*-

import tensorflow as tf


def center_size_bbox_to_corners_bbox(pred_bbox):
    x_min = tf.maximum(0, pred_bbox[:, 0] - 0.5 * pred_bbox[:, 2])
    y_min = tf.maximum(0, pred_bbox[:, 1] - 0.5 * pred_bbox[:, 3])
    x_max = tf.maximum(0, pred_bbox[:, 0] + pred_bbox[:, 2])
    y_max = tf.maximum(0, pred_bbox[:, 1] + pred_bbox[:, 3])

    return tf.concat([x_min, y_min, x_max, y_max], axis=1)


def area(boxlist, scope=None):
    """Computes area of boxes.

      Args:
        boxlist: BoxList holding N boxes
        scope: name scope.

      Returns:
        a tensor with shape [N] representing box areas.
    """
    with tf.name_scope(scope, 'Area'):
        y_min, x_min, y_max, x_max = tf.split(value=boxlist, num_or_size_splits=4, axis=1)
        return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])


def intersection_area(pred_boxes, gt_boxes, scope=None):
    """Compute pairwise intersection areas between boxes.

      Args:
        boxlist1: BoxList holding N boxes
        boxlist2: BoxList holding M boxes
        scope: name scope.

      Returns:
        a tensor with shape [N, M] representing pairwise intersections
   """
    with tf.name_scope(scope, 'Intersection'):
        y_min1, x_min1, y_max1, x_max1 = tf.split(value=pred_boxes, num_or_size_splits=4, axis=1)
        y_min2, x_min2, y_max2, x_max2 = tf.split(value=gt_boxes, num_or_size_splits=4, axis=1)

        all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
        all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))
        all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
        all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))

        intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
        intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)

        return intersect_heights * intersect_widths


def iou_per_iamge(boxlist1, boxlist2, scope=None):
    """Computes pairwise intersection-over-union between box collections.

      Args:
        boxlist1: BoxList holding N boxes
        boxlist2: BoxList holding M boxes
        scope: name scope.

      Returns:
        two tensor with shape [N,] representing per box max iou scores
        and corresponding index(use to find gt_class)
    """
    with tf.name_scope(scope, 'IOU'):
        intersections = intersection_area(boxlist1, boxlist2)
        areas1 = area(boxlist1)
        areas2 = area(boxlist2)

        # N 个areas1, M 个 areas2, 通过这种笛卡尔积的方式相加, 从而得到 N×M的矩阵
        unions = (tf.expand_dims(areas1, 1) +
                  tf.expand_dims(areas2, 0) - intersections)
        iou = tf.truediv(intersections, unions)

        #  真实标签(gt_bbox_list, gt_class_list)的下标
        gt_list_index = tf.argmax(iou, axis=1)  # per box
        max_iou = tf.reduce_max(iou, axis=1)  # per box
        return gt_list_index, max_iou


def responsible_box_per_image(pred_bbox, gt_bbox):
    """用于找出一张图片每个 cell 预测的两个 bbox 中用于负责预测的 bbox, pred_bbox shape=[N, 4]

    Args:
        pred_bbox: 预测 bbox, 维度为 7x7x8
        gt_bbox: 真实 bbox(num_object x 4)

    Returns:
         用于负责的预测的 bbox, 以及对应类信息
         responsible_box_index: 每个 bbox 与 gt-bboxes 有最大 Iou 的 gt-bbox 的在
                                gt-boxer list中索引信息
    """
    class_index, confidence = iou_per_iamge(pred_bbox, gt_bbox)
    reshaped_confidence = tf.reshape(confidence, [-1, 2])
    reshaped_class_index = tf.reshape(class_index, [-1, 2])
    responsible_box_index = tf.argmax(reshaped_confidence, axis=1)  # per cell
    responsible_box_iou = tf.reduce_max(reshaped_confidence, axis=1) # per cell

    return reshaped_class_index, responsible_box_iou, responsible_box_index


def batch_loss(predictions, gt_boxes_lists, gt_class_lists, batch_size=64):

    locate_losses, iou_losses, cls_losses = 0, 0, 0
    for i in range(batch_size):
        prediction = predictions[i, :, :, :]
        prediction = tf.reshape(prediction, [-1, 30])
        gt_boxes_list = tf.squeeze(gt_boxes_lists[i, -1, 4])
        gt_class_list = tf.squeeze(gt_class_lists[i, -1])
        locate_loss, iou_loss, cls_loss = per_image_loss(prediction,
                                                         gt_boxes_list,
                                                         gt_class_list)
        locate_losses += locate_loss
        iou_losses += iou_loss
        cls_losses += cls_loss

    tf.summary.scalar("losses/locate_loss", locate_losses)
    tf.summary.scalar("losses/iou_loss", iou_losses)
    tf.summary.scalar("losses/cls_loss", cls_losses)

    total_loss = 5 * locate_losses + iou_losses + cls_losses

    return total_loss


def per_image_loss(prediction, gt_boxes_list, gt_class_list):
    """
    Args:
         prediction: reshape 为 [49, 30] 的最后一层 feature map
         gt_boxes_list: 图片真实 bbox
         gt_class_list: 图片中每个物体的类别

    Return: 损失
    """
    pred_bbox = prediction[:, 0:8]
    pred_iou = prediction[:, 8:10]
    pred_class = prediction[:, 10:30]

    reshaped_pred_bbox = tf.reshape(pred_bbox, [-1, 4])
    class_index, responsible_box_iou, responsible_box_index = \
        responsible_box_per_image(reshaped_pred_bbox, gt_boxes_list)

    locate_loss = 0
    iou_loss = 0
    cls_loss = 0

    # 一张图片一个一个格子的处理, 哎, 这效率:(
    for idx in tf.range(reshaped_pred_bbox.shape[0]):
        box_index_per_cell = responsible_box_index[idx]
        if box_index_per_cell == 0:
            responsible_box = pred_bbox[idx, 0:4]
        else:
            responsible_box = pred_bbox[idx, 4:8]

        gt_box = gt_boxes_list[class_index[idx, box_index_per_cell]]
        locate_loss += tf.square(responsible_box[0:2] - gt_box[0:2]) + \
                    tf.square(tf.sqrt(responsible_box[2:4]) - tf.sqrt(gt_box[2:4]))

        if responsible_box_iou < 0.001:
            iou_loss += 0.5 * tf.square(responsible_box_iou - pred_iou[box_index_per_cell])
        else:
            iou_loss += tf.square(responsible_box_iou - pred_iou[box_index_per_cell])
            cls_loss += tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
                labels=gt_class_list[idx, box_index_per_cell],
                logits=pred_class(idx)))

    return locate_loss, iou_loss, cls_loss
