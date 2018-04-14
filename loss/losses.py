# -*-coding:utf-8-*-

import tensorflow as tf
import numpy as np

CELL_SIZE = 7
BOX_PER_CELL = 2
MAX_NUM_OBJECT = 20


def center_size_bbox_to_corners_bbox(bboxlist, axis=-1):
    x_center, y_center, width, height = tf.split(value=bboxlist, num_or_size_splits=4, axis=axis)
    x_min = tf.maximum(0.0, x_center - 0.5 * width)
    y_min = tf.maximum(0.0, x_center - 0.5 * height)
    x_max = tf.maximum(0.0, x_center + 0.5 * width)
    y_max = tf.maximum(0.0, x_center + 0.5 * height)

    return tf.concat([x_min, y_min, x_max, y_max], axis=axis)


def corners_bbox_to_center_bbox(bboxlist, axis=-1):
    x_min, y_min, x_max, y_max = tf.split(value=bboxlist, num_or_size_splits=4, axis=axis)
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    width = x_max - x_min
    height = y_max - y_min

    return tf.concat([x_center, y_center, width, height], axis=axis)


def area(boxlist, axis=-1):
    """Computes area of boxes.

      Args:
        boxlist: BoxList holding N boxes
        scope: name scope.

      Returns:
        a tensor with shape [N] representing box areas.
    """
    with tf.name_scope('Area'):
        x_min, y_min, x_max, y_max = tf.split(value=boxlist, num_or_size_splits=4, axis=axis)
        heights = tf.maximum(0.0, y_max - y_min)
        widths = tf.maximum(0.0, x_max - x_min)
        return tf.squeeze(heights * widths, axis=axis)


def intersection(pred_boxes, gt_boxes, axis=-1):
    """Compute pairwise intersection areas between boxes.

      Args:
        pred_boxes: BoxList holding N boxes [N:4] (xmin, ymin, xmax, ymax)
        gt_boxes: BoxList holding M boxes [M, 4],  (xmin, ymin, xmax, ymax)
        axis: xx

      Returns:
        a tensor with shape [N, M] representing pairwise intersections
   """
    with tf.name_scope('Intersection'):
        x_min1, y_min1, x_max1, y_max1 = tf.split(value=pred_boxes, num_or_size_splits=4, axis=axis)
        x_min2, y_min2, x_max2, y_max2 = tf.split(value=gt_boxes, num_or_size_splits=4, axis=axis)

        max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
        max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))

        min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
        min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))

        return tf.stack([max_xmin, max_ymin, min_xmax, min_ymax], axis=axis)


def iou_per_image(pred_bbox, gt_bbox):
    """Computes pairwise intersection-over-union between box collections.

      Args:
        pred_bbox: BoxList holding N boxes [N, 4] (xmin, ymin, xmax, ymax)
        gt_bbox: BoxList holding M boxes [M. 4] (xmin, ymin, xmax, ymax)
        scope: name scope.

      Returns:
        two tensor with shape [N,] representing per box max iou scores
        and corresponding index(use to find gt_class)
    """
    with tf.name_scope('IOU'):
        intersect = intersection(pred_bbox, gt_bbox)
        print("intersect shape ", intersect.shape)
        intersect_area = area(intersect)
        areas1 = area(pred_bbox)
        areas2 = area(gt_bbox)

        # N 个areas1, M 个 areas2, 通过这种笛卡尔积的方式相加, 从而得到 N×M 的矩阵
        union_area = tf.expand_dims(areas1, -1) + tf.expand_dims(areas2, 0) - intersect_area + 1e-5
        iou = tf.truediv(intersect_area, union_area)

        return iou


def per_image_loss(pred, gt_bbox, gt_class):
    """ 计算一张图片对应的 loss

    Args:
        pred: 预测 bbox, 维度为 7x7x30
        gt_bbox: 真实 bbox(num_object x 4)
        gt_class: 真实 class(num_object)

    Returns: xx
    """
    center_gt_bbox = corners_bbox_to_center_bbox(gt_bbox)  # center_bbox
    print(center_gt_bbox.shape)
    x_center = tf.floor(center_gt_bbox[:, 0] * CELL_SIZE)
    y_center = tf.floor(center_gt_bbox[:, 1] * CELL_SIZE)

    pred_bbox = pred[:, :, 0:8]
    pred_bbox = tf.reshape(pred_bbox, [CELL_SIZE, CELL_SIZE, 2, 4])
    pred_iou = pred[:, :, 8:10]
    pred_iou = tf.reshape(pred_iou, [CELL_SIZE, CELL_SIZE, 2])
    pred_class = pred_bbox[:, :, 10:]

    # 由于 tensor 不能直接对某个元素赋值, 所以采用这种方法
    base_boxes = np.zeros([CELL_SIZE, CELL_SIZE, 4], dtype=np.float32)

    for x in range(CELL_SIZE):
        for y in range(CELL_SIZE):
            base_boxes[x, y, 0] = x / CELL_SIZE
            base_boxes[x, y, 1] = y / CELL_SIZE
    base_boxes = np.reshape(base_boxes, [CELL_SIZE, CELL_SIZE, 1, 4])
    base_boxes = np.tile(base_boxes, [1, 1, BOX_PER_CELL, 1])

    # 将相对于每个 cell left-bottom 的坐标 (x, y), 转化为相对于图片的 left-bottom 的坐标
    pred_bbox = tf.concat([pred_bbox[:, :, :, 0:2]/CELL_SIZE, pred_bbox[:, :, :, 2:]], axis=-1)
    pred_bbox = pred_bbox + base_boxes

    corner_pred_bbox = center_size_bbox_to_corners_bbox(pred_bbox, axis=-1)
    print("corner_pred_bbox",  corner_pred_bbox.shape)
    iou = iou_per_image(corner_pred_bbox, gt_bbox)
    print("iou shape", iou.shape)

    class_loss = tf.Variable(0, tf.float32)
    object_iou_loss = tf.Variable(0, tf.float32)
    no_object_iou_loss = tf.Variable(0, tf.float32)
    coord_loss = tf.Variable(0, tf.float32)
    mask = np.ones(shape=[CELL_SIZE, CELL_SIZE], dtype=np.float32)

    for idx in range(MAX_NUM_OBJECT):

        try:
            x = x_center[idx]
            y = y_center[idx]
            mask[x, y] = 0.0
            responsible_box_iou = tf.reduce_max(iou[x, y, :, idx])
            responsible_box_index = tf.argmax(iou[x, y, :, idx])
            object_iou_loss += tf.reduce_mean(tf.square(responsible_box_iou - pred_iou[x, y, responsible_box_index]))
            one_hot_label = tf.one_hot(gt_class[idx], depth=20, dtype=tf.float32)
            class_loss += tf.reduce_mean(tf.square(one_hot_label - pred_class[x, y]))
            coord_loss += 5.0 * tf.reduce_mean((tf.square(gt_bbox[idx, 0:2] - pred_bbox[x, y, responsible_box_index, 0:2]) +
                               tf.square(tf.sqrt(gt_bbox[idx, 2:4]) - tf.sqrt(pred_bbox[x, y, responsible_box_index, 2:4]))))
        except IndexError:
            break

    responsible_cell_iou = tf.reduce_max(iou, axis=-1)
    responsible_box_iou = tf.reduce_max(responsible_cell_iou, axis=-1, keepdims=True)
    responsible_box_index = tf.cast(responsible_cell_iou >= responsible_box_iou, tf.float32)
    print("responsible_box_index", responsible_box_index.shape)

    mask = np.reshape(mask, [CELL_SIZE, CELL_SIZE, 1])

    no_object_pred_iou = responsible_box_index * responsible_box_iou * mask
    print(no_object_pred_iou.shape)
    no_object_iou_loss += 0.5 * tf.reduce_mean(tf.square(no_object_pred_iou))

    return coord_loss, object_iou_loss, no_object_pred_iou, class_loss


def batch_loss(predictions, gt_boxes, gt_class, batch_size=1):
    class_loss = tf.Variable(0.0, tf.float32)
    object_iou_loss = tf.Variable(0.0, tf.float32)
    no_object_iou_loss = tf.Variable(0.0, tf.float32)
    coord_loss = tf.Variable(0.0, tf.float32)

    for i in range(batch_size):
        prediction = predictions[i, :, :, :]
        gt_boxes = gt_boxes[i, :, :]
        gt_class = gt_class[i, :]
        coord_loss1, object_iou_loss1, no_object_iou_loss1, class_loss1 \
            = per_image_loss(prediction, gt_boxes, gt_class)
        coord_loss += coord_loss1
        object_iou_loss += object_iou_loss1
        no_object_iou_loss += no_object_iou_loss1
        class_loss += class_loss1

    yolo_loss = coord_loss + object_iou_loss + no_object_iou_loss + class_loss

    tf.summary.scalar("losses/locate_loss", coord_loss)
    tf.summary.scalar("losses/class_loss", class_loss)
    tf.summary.scalar("losses/object_iou_loss", object_iou_loss)
    tf.summary.scalar("losses/no_object_iou_loss", no_object_iou_loss)
    tf.summary.scalar("losses/yolo_loss", yolo_loss)

    return yolo_loss
