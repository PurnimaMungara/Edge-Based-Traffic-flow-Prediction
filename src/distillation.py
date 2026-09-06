import tensorflow as tf


class Distiller(tf.keras.Model):
    """
    Knowledge Distillation model for regression.

    The teacher is a larger model and the student is a smaller model.
    The student learns from:
        1. Ground-truth traffic values
        2. Teacher predictions
    """

    def __init__(self, student, teacher, alpha=0.5):
        super().__init__()

        self.teacher = teacher
        self.student = student
        self.alpha = alpha

        # Teacher should not be trained during distillation.
        self.teacher.trainable = False

        self.student_loss_fn = tf.keras.losses.MeanSquaredError()
        self.distillation_loss_fn = tf.keras.losses.MeanSquaredError()

        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.student_loss_tracker = tf.keras.metrics.Mean(
            name="student_loss"
        )
        self.distillation_loss_tracker = tf.keras.metrics.Mean(
            name="distillation_loss"
        )
        self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(
            name="mae"
        )

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.student_loss_tracker,
            self.distillation_loss_tracker,
            self.mae_tracker,
        ]

    def call(self, inputs, training=False):
        """
        IMPORTANT:
        Keras 3 requires custom Model classes to implement call().
        The Distiller returns the student's prediction.
        """
        return self.student(inputs, training=training)

    def compile(self, optimizer, **kwargs):
        """
        Compile the student/distillation model.
        """
        super().compile(
            optimizer=optimizer,
            **kwargs
        )

    def train_step(self, data):
        """
        One training step for knowledge distillation.
        """

        # Support both:
        # (x, y)
        # and
        # ((x,), y)
        if isinstance(data, tuple):
            x, y = data
        else:
            x = data[0]
            y = data[1]

        # Teacher prediction.
        # Teacher must not receive gradients.
        teacher_predictions = self.teacher(
            x,
            training=False
        )

        with tf.GradientTape() as tape:

            # Student prediction.
            student_predictions = self.student(
                x,
                training=True
            )

            # Student error against real traffic values.
            student_loss = self.student_loss_fn(
                y,
                student_predictions
            )

            # Distillation error:
            # Student learns to imitate teacher.
            distillation_loss = self.distillation_loss_fn(
                teacher_predictions,
                student_predictions
            )

            # Total loss.
            #
            # alpha = 0.5 means:
            # 50% real data loss
            # 50% teacher imitation loss
            total_loss = (
                self.alpha * student_loss
                + (1.0 - self.alpha) * distillation_loss
            )

        # Only update student model.
        gradients = tape.gradient(
            total_loss,
            self.student.trainable_variables
        )

        self.optimizer.apply_gradients(
            zip(
                gradients,
                self.student.trainable_variables
            )
        )

        # Update metrics.
        self.total_loss_tracker.update_state(total_loss)
        self.student_loss_tracker.update_state(student_loss)
        self.distillation_loss_tracker.update_state(
            distillation_loss
        )
        self.mae_tracker.update_state(
            y,
            student_predictions
        )

        return {
            "loss": self.total_loss_tracker.result(),
            "student_loss": self.student_loss_tracker.result(),
            "distillation_loss": self.distillation_loss_tracker.result(),
            "mae": self.mae_tracker.result(),
        }

    def test_step(self, data):
        """
        Validation/test step.
        """

        if isinstance(data, tuple):
            x, y = data
        else:
            x = data[0]
            y = data[1]

        teacher_predictions = self.teacher(
            x,
            training=False
        )

        student_predictions = self.student(
            x,
            training=False
        )

        student_loss = self.student_loss_fn(
            y,
            student_predictions
        )

        distillation_loss = self.distillation_loss_fn(
            teacher_predictions,
            student_predictions
        )

        total_loss = (
            self.alpha * student_loss
            + (1.0 - self.alpha) * distillation_loss
        )

        self.total_loss_tracker.update_state(total_loss)
        self.student_loss_tracker.update_state(student_loss)
        self.distillation_loss_tracker.update_state(
            distillation_loss
        )
        self.mae_tracker.update_state(
            y,
            student_predictions
        )

        return {
            "loss": self.total_loss_tracker.result(),
            "student_loss": self.student_loss_tracker.result(),
            "distillation_loss": self.distillation_loss_tracker.result(),
            "mae": self.mae_tracker.result(),
        }