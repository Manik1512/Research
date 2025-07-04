import tensorflow as tf

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1_score", **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name="true_positives", initializer="zeros")
        self.fp = self.add_weight(name="false_positives", initializer="zeros")
        self.fn = self.add_weight(name="false_negatives", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > 0.5, tf.float32)  # Convert probabilities to binary
        y_true = tf.cast(y_true, tf.float32)

        self.tp.assign_add(tf.reduce_sum(y_true * y_pred))
        self.fp.assign_add(tf.reduce_sum((1 - y_true) * y_pred))
        self.fn.assign_add(tf.reduce_sum(y_true * (1 - y_pred)))

    def result(self):
        precision = self.tp / (self.tp + self.fp + tf.keras.backend.epsilon())
        recall = self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())
        f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        return f1

    def reset_state(self):
        self.tp.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)


class Accuracy(tf.keras.metrics.Metric):
    def __init__(self, name="accuracy", **kwargs):
        super(Accuracy, self).__init__(name=name, **kwargs)
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        correct_preds = tf.reduce_sum(tf.cast(y_pred == y_true, tf.float32))
        total_preds = tf.cast(tf.size(y_true), tf.float32)

        self.correct.assign_add(correct_preds)
        self.total.assign_add(total_preds)

    def result(self):
        return self.correct / self.total

    def reset_state(self):
        self.correct.assign(0)
        self.total.assign(0)

class AUC(tf.keras.metrics.AUC):
    def __init__(self, name="auc", **kwargs):
        super(AUC, self).__init__(name=name, **kwargs)


class MCC(tf.keras.metrics.Metric):
    def __init__(self, name="mcc", **kwargs):
        super(MCC, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name="true_positives", initializer="zeros")
        self.tn = self.add_weight(name="true_negatives", initializer="zeros")
        self.fp = self.add_weight(name="false_positives", initializer="zeros")
        self.fn = self.add_weight(name="false_negatives", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        self.tp.assign_add(tf.reduce_sum(y_true * y_pred))
        self.tn.assign_add(tf.reduce_sum((1 - y_true) * (1 - y_pred)))
        self.fp.assign_add(tf.reduce_sum((1 - y_true) * y_pred))
        self.fn.assign_add(tf.reduce_sum(y_true * (1 - y_pred)))

    def result(self):
        numerator = (self.tp * self.tn) - (self.fp * self.fn)
        denominator = tf.sqrt(
            (self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn)
        )
        return numerator / (denominator + tf.keras.backend.epsilon())

    def reset_state(self):
        self.tp.assign(0)
        self.tn.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)


def dice_loss(y_true, y_pred, epsilon=1e-7):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])

    denominator = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3]) + epsilon

    dice_coefficient = numerator / denominator
    loss = 1 - dice_coefficient
    return tf.reduce_mean(loss)

from focal_loss import BinaryFocalLoss

fl=BinaryFocalLoss(gamma=3)

def fused_loss(y_true, y_pred, kappa=1.0):
    dice = dice_loss(y_true, y_pred)
    focal = fl(y_true, y_pred)
    total_loss = dice + kappa * focal
    return total_loss

    